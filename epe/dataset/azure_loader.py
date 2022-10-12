# %%
import logging
from operator import contains
import os
from datetime import datetime, timedelta
from io import BytesIO
from random import uniform
from time import sleep
from typing import Optional, Union

import azure.identity
import azure.identity.aio
import numpy as np
import requests
from aiohttp import ServerTimeoutError
from azure.core.credentials import AccessToken
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError, ResourceNotFoundError, \
    ServiceRequestError
from azure.storage.blob import BlobServiceClient
from requests.exceptions import RequestException
from retry import retry
from PIL import Image


TEACHER_SEMSEG = 'segmentation/panoptic_deeplab_960x448_f=360'
TEACHER_DEPTH = 'depth/efficientnet-b2-ssd-depthnet_960x448_f=360'
TEACHER_CUBOID = 'cuboids/fcos3d'

MAX_AUTH_ENDPOINT_ATTEMPTS = 30

# HACK: Azure SDK doesn't consistently refresh credentials, so training can
# fail after 24 hours. We make sure that new tokens get issued semi-regularly
# to avoid this issue.
# See https://app.clubhouse.io/wayve/story/14774/azure-auth-error-during-training
# for details.
FORCE_INSTANCE_RECREATE_AFTER_MINUTES = 59

AZURE_SHARDS = 8  # number of storage container shards in Azure blob storage


# %%
class AzureCredentials:
    """
    A class that obtains and caches Azure credentials. AzureCredentials().get_azure_credential() will try
    to obtain credentials using the following logic:
     1. If AZURE_CLIENT_ID is set, use EnvironmentCredential (Via service principal)
     2. If IS_RUNNING_IN_AZURE is true use ManagedIdentityCredential
     3. If AZURE_ACCOUNT_KEY is set, return this as a string
     4. Else tries AzureCliCredential
    This class also contains some workarounds to deal with timeouts and other
    exception cases when running in multiple processes.
    """

    _instance: Optional['AzureCredentials'] = None
    _azure_identity = None
    _created_time: Optional[datetime] = None
    use_async: Optional[bool] = None

    def __new__(cls, use_async=False) -> 'AzureCredentials':
        if cls._instance is None or cls._should_recreate() or use_async != cls.use_async:
            cls.use_async = use_async
            cls._recreate()

        # help type-checking know that cls._instance cannot be None at this point by
        # doing an assertion
        assert cls._instance is not None
        return cls._instance

    @classmethod
    def _recreate(cls):
        # Assuming training won't be run in Azure for now to avoid WayveCode dependency
        # from wayve.services.common.environment_detection import IS_RUNNING_IN_AZURE
        IS_RUNNING_IN_AZURE = False

        cls._instance = super(AzureCredentials, cls).__new__(cls)
        cls._azure_identity = None

        if os.environ.get('AZURE_USE_DEVICE_CODE'):
            cls._azure_credential_cached_device_code()
        elif os.environ.get('AZURE_CLIENT_ID'):
            cls._azure_credential_environment()
        elif IS_RUNNING_IN_AZURE:
            cls._azure_authentication_in_azure()
        elif os.environ.get('AZURE_ACCOUNT_KEY'):
            cls._azure_credential_account_key()
        else:
            cls._azure_cli_authentication()

        cls._created_time = datetime.utcnow()

    @classmethod
    def _should_recreate(cls):
        if not FORCE_INSTANCE_RECREATE_AFTER_MINUTES:
            return False

        if not cls._created_time:
            return True

        now = datetime.utcnow()
        delta = timedelta(minutes=FORCE_INSTANCE_RECREATE_AFTER_MINUTES)
        should_refresh = cls._created_time < now - delta
        return should_refresh

    @classmethod
    def _azure_credential_cached_device_code(cls) -> None:
        # use_async will only be able to use the cached credentials, not manage the Device Code flow
        tcpo = azure.identity.TokenCachePersistenceOptions(allow_unencrypted_storage=True)

        if cls.use_async:
            cls._azure_identity = azure.identity.aio.SharedTokenCacheCredential(cache_persistence_options=tcpo)
        else:
            cls._azure_identity = azure.identity.ChainedTokenCredential(
                azure.identity.SharedTokenCacheCredential(cache_persistence_options=tcpo),
                azure.identity.DeviceCodeCredential(cache_persistence_options=tcpo),
            )

    @classmethod
    def _azure_credential_environment(cls) -> None:
        identity = azure.identity.aio if cls.use_async else azure.identity
        # Attempt EnvironmentCredential (authenticating via a Service Principal)
        cls._azure_identity = identity.EnvironmentCredential()

    @classmethod
    def _azure_authentication_in_azure(cls):
        identity = azure.identity.aio if cls.use_async else azure.identity
        managed_identity_client_id = os.environ.get('MANAGED_IDENTITY_CLIENT_ID')
        cls._managed_identity_prep()
        if managed_identity_client_id:
            cls._azure_identity = identity.ManagedIdentityCredential(client_id=managed_identity_client_id)
        else:
            cls._azure_identity = identity.ManagedIdentityCredential()

    @classmethod
    def _azure_credential_account_key(cls) -> Optional[str]:
        cls._azure_identity = os.environ.get('AZURE_ACCOUNT_KEY')
        return cls._azure_identity

    @classmethod
    def _azure_cli_authentication(cls):
        identity = azure.identity.aio if cls.use_async else azure.identity

        cls._azure_identity = identity.AzureCliCredential()

    @classmethod
    @retry(
        exceptions=requests.RequestException,
        tries=MAX_AUTH_ENDPOINT_ATTEMPTS,
        delay=0.1,
        logger=logging.getLogger(__name__),
    )
    def _wait_for_auth_endpoint(cls, timeout=1) -> None:
        # When running within AzureContainerInstance, Managed Identity auth endpoint
        # isn't available in the first few seconds. Wait until auth endpoint becomes available.
        requests.get('http://169.254.169.254/metadata/identity/oauth2/token', timeout=timeout)

    @classmethod
    def _managed_identity_prep(cls) -> None:
        # Hack around Azure limitations: sleep a random amount of time
        # so that when we have 48+ threads all calling the Managed Identity
        # endpoint at the start of training, we do not get rate-limited.
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        sleep_time_s = max(5, int(world_size / 2))  # Expecting 100ms per worker thread and 4+ workers per rank
        sleep(uniform(0, sleep_time_s))

        try:
            cls._wait_for_auth_endpoint()
        except RequestException:
            raise RuntimeError('Failed to reach Managed Identity Endpoint')

    @classmethod
    def get_azure_credential(cls) -> Union[str, AccessToken]:
        if cls._azure_identity:
            return cls._azure_identity

        raise RuntimeError('Failed all authentication methods')

def image_match_desired_size(img, height_new, width_new):
    width, height = img.size

    old_aspect = width / height
    new_aspect = width_new / height_new

    # rescale on smaller dimension (aspect ratio wise)
    # then crop excess
    if new_aspect > old_aspect:
        scale = width_new / width
        height = int(scale * height)
        width = width_new
    else:
        scale = height_new / height
        height = height_new
        width = int(scale * width)

    # resizing to match desired height
    img = img.resize((width, height))

    # cropping to match desired width
    center_x = width // 2
    left = center_x - width_new//2
    right = center_x + width_new//2

    center_y = height // 2
    top = center_y - height_new//2
    bottom = center_y + height_new//2
    img = img.crop((left, top, right, bottom))

    return img


# %%
class AzureImageLoader:
    def __init__(self):
        self.shards = AZURE_SHARDS

        self.initialize(-2)  # sim images
        self.initialize(-1)  # cold storage
        self.hot_services = [None] * self.shards  # hot storage
        for i in range(self.shards):
            self.initialize(i)
        print("Azure Image Loader initialized")
        
        logging.getLogger("azure.storage.common.storageclient").setLevel(logging.WARNING)
        logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.WARNING)


    def initialize(self, i):
        if i >= 0:
            # hot storage
            account_name = 'wayveprodtraining%02d' % (i + 1)
            self.hot_services[i] = BlobServiceClient(
                account_url='https://%s.blob.core.windows.net' % account_name,
                credential=AzureCredentials(use_async=False).get_azure_credential(),
                read_timeout=60,
            )
        elif i == -1:
            # cold storage
            account_name = 'wayveingest'
            self.cold_service = BlobServiceClient(
                account_url='https://%s.blob.core.windows.net' % account_name,
                credential=AzureCredentials(use_async=False).get_azure_credential(),
                read_timeout=60,
            )
        elif i == -2:
            # sim storage
            account_name = 'wayveprodsimingest'
            self.sim_service = BlobServiceClient(
                account_url='https://%s.blob.core.windows.net' % account_name,
                credential=AzureCredentials(use_async=False).get_azure_credential(),
                read_timeout=60,
            )
        else:
            raise NotImplementedError(str(i))
        # print('initialized BlobServiceClient: %s'%account_name)

    def load_img_from_path(self, path):
        path = str(path).split('/')
        run_id = '/'.join(path[:2])
        camera = path[3]

        mode = 'rgb'
        if 'ningaloo' in run_id:
            camera_split = camera.split('--')
            camera = camera_split[0]
            mode = camera_split[1]

        name = path[4]
        # TODO: use a cleaner way of obtaining the int
        timestamp = int(name.replace('unixus', '')
            .replace('.jpeg','').replace('.png', ''))
        output = self.load(run_id, camera, timestamp, mode=mode) 
        return Image.open(output)
            

    def load_img_from_path_and_resize(self, path, height, width):
        img = self.load_img_from_path(path)
        return image_match_desired_size(img, height, width)

    def load_sim_img_from_path(self, path):
        blob_client = self.sim_service.get_blob_client(path)
        try:
            if blob_client.exists():  # TODO: this extra check is a bit slow but prevents a memory leak (when using ddp+workers)
                data = blob_client.download_blob().readall()
                output = BytesIO(data)
                return output
            else:
                raise ResourceNotFoundError

        except ResourceNotFoundError:
            print('image not in sim storage: ', path)
            return None

    def load(self, run_id, camera, timestamp, mode='rgb', calibration_hash=None, hot=True, auth_attempts=1) -> BytesIO:
        if run_id[:8] == 'ningaloo':
            hot = False
            shard_index = -2
            ext_map = {'rgb': 'jpeg', 'segmentation': 'png', 'depth': 'png'}
            blob_name = f'{run_id}/cameras/{camera}--{mode}/{timestamp:012d}unixus.{ext_map[mode]}'
            blob_client = self.sim_service.get_blob_client('images', blob_name)
        elif hot:
            timestamp_sec = int(timestamp / 1e6)
            shard_index = timestamp_sec % self.shards
            if mode == 'rgb':
                container = 'image-cache'
                full_name = f'jpeg-full_resolution/{run_id}/cameras/{camera}/{timestamp:012d}unixus.jpeg'
            elif mode == 'segmentation':
                # assert calibration_hash is not None
                container = 'teacher-cache'
                full_name = f'{TEACHER_SEMSEG}/{run_id}/{camera}/{calibration_hash}/{timestamp:012d}unixus.png'
            elif mode == 'depth':
                container = 'teacher-cache'
                full_name = f'{TEACHER_DEPTH}/{run_id}/{camera}/{calibration_hash}/{timestamp:012d}unixus.png'
            elif mode == 'cuboids':
                container = 'teacher-cache'
                full_name = f'{TEACHER_CUBOID}/json-1920x1200/{run_id}/cameras/{camera}/{timestamp:012d}unixus.json'
            else:
                raise NotImplementedError(mode)
            

            blob_client = self.hot_services[shard_index].get_blob_client(container, full_name)
        else:
            shard_index = -1
            blob_name = f'{run_id}/cameras/{camera}/{timestamp:012d}unixus.jpeg'
            blob_client = self.cold_service.get_blob_client('vehicle-data', blob_name)
        try:
            if blob_client.exists():  # TODO: this extra check is a bit slow but prevents a memory leak (when using ddp+workers)
                data = blob_client.download_blob().readall()
                if mode in ['rgb', 'segmentation', 'depth']:
                    output = BytesIO(data)  # this needs to be in try block in case data is corrupted
                else:
                    output = BytesIO(data)
            else:
                raise ResourceNotFoundError

        except ResourceNotFoundError:
            # if not in hot cache try to load from cold storage
            if hot and mode == 'rgb':
                return self.load(run_id, camera, timestamp, mode, calibration_hash, False)
            else:
                print('image not in hot nor cold storage:', run_id, camera, timestamp, mode)
                return None

        except (ServiceRequestError, ServerTimeoutError, HttpResponseError) as e:

            print(e)
            return None

        except ClientAuthenticationError:
            print('ClientAuthenticationError... attempting to reauth')
            if auth_attempts > 0:
                self.initialize(shard_index)
                return self.load(run_id, camera, timestamp, mode, calibration_hash, hot, auth_attempts - 1)
            print('failed to reauth!!!')
            raise

        return output

    def __call__(self, run_id, camera, timestamp, mode='image', calibration_hash=None):
        return self.load(run_id, camera, timestamp, mode, calibration_hash, True)

    def get_timestamps(self, path):
        is_sim = "ningaloo" in path
        service = self.sim_service if is_sim else self.cold_service

        blobs_container = "images" if is_sim else "vehicle-data"

        timestamps = []
        for blob in service.get_container_client(blobs_container).list_blobs(path):
            name = os.path.splitext(os.path.split(blob.name)[1])[0]
            if name[-6:] == 'unixus':
                timestamps.append(int(name[:-6]))
            # TODO: only gettig one timestamp for faster debugging purposes
            break
        return np.array(timestamps)

    def get_image_timestamps(self, runid, camera):
        # print("getting timestamps")
        return self.get_timestamps(os.path.join(runid, 'cameras', camera))

# %%

# %%
