# %%
import logging
from operator import contains
import os
from datetime import datetime, timedelta
from io import BytesIO
from random import uniform
from threading import local
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
from tqdm import tqdm
from epe.dataset.utils import read_azure_filelist


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

# %%
class AzureImageLoader:
    def __init__(self):
        self.shards = AZURE_SHARDS

        self.initialize() 
        
        logging.getLogger("azure.storage.common.storageclient").setLevel(logging.WARNING)
        logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.WARNING)


    def initialize(self):
        # sim storage
        account_name = 'wayveprodsimingest'
        self.sim_service = BlobServiceClient(
            account_url='https://%s.blob.core.windows.net' % account_name,
            credential=AzureCredentials(use_async=False).get_azure_credential(),
            read_timeout=60,
        )
            
    def upload_img(self, path, dataset_name, local_data_root='/mnt/remote/data/users/kacper/datasets'):
        upload_file_path = os.path.join(local_data_root, path)
        path = os.path.join(path)
        blob_client = self.sim_service.get_blob_client('sim2real-image-translation', path)
        if not blob_client.exists():
            with open(upload_file_path, "rb") as data:
                try:
                    blob_client.upload_blob(data, overwrite=False)
                except:
                    print('skipping')

    def read_img(self, path, dataset_name, local_data_root='/mnt/remote/data/users/kacper/datasets'):
        print(path)
        print(local_data_root)
        upload_file_path = os.path.join(local_data_root, path)
        print('\n')
        print(upload_file_path)
        img =  Image.open(upload_file_path)
        print(img.size)
        path = os.path.join(path)

        blob_client = self.sim_service.get_blob_client('sim2real-image-translation', path)
        if not blob_client.exists():
            with open(upload_file_path, "rb") as data:
                try:
                    blob_client.upload_blob(data, overwrite=False)
                except:
                    print('skipping')



# %%
loader = AzureImageLoader()
dataset_name = 'urban-driving'
dataset_root = '/home/kacper/data/datasets/urban-driving'

def upload_files(files, dataset_name='somers-town_weather_v1'):
    # files: tuple of files to be uploaded (e.g. sharing timestamp)
    for ts in tqdm(files):
        for file in ts:
            # loader.upload_img(file, dataset_name=dataset_name, local_data_root=dataset_root)
            loader.read_img(file, dataset_name=dataset_name, local_data_root='/home/kacper/data/datasets')
        break

# %%
# print('uploading sim')
sim_files = read_azure_filelist(os.path.join(dataset_root, 'sim_files.csv'), ['rgb', 'depth', 'normal', 'segmentation'], dataset_name=dataset_name)
upload_files(sim_files, dataset_name=dataset_name)


# %%
print('uploading real')
real_files = read_azure_filelist(os.path.join(dataset_root, 'real_files.csv'), ['rgb', 'segmentation'], dataset_name=dataset_name)
upload_files(real_files, dataset_name=dataset_name)

# %%
