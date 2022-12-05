from io import BytesIO

import numpy as np
from PIL import Image

PAD_VALUE_DEPTH_UINT16 = 65535
PAD_VALUE_DEPTH_FLOAT32 = -1.0		

_DEPTH_NORM_CONST = 2 ** 16 - 1
MIN_DEPTH_M = 0.25
MAX_DEPTH_M = 1000
MAX_INVDEPTH_INVM = 1 / MIN_DEPTH_M
MIN_INVDEPTH_INVM = 1 / MAX_DEPTH_M

MAX_NORMAL = 255

def is_jpeg_encoded(np_img: np.ndarray) -> bool:
    img_shape = np_img.shape
    if len(img_shape) != 1:
        return False
    return np_img[0] == 0xFF and np_img[1] == 0xD8 and np_img[2] == 0xFF and np_img[3] == 0xE0


def np_jpeg_bytes_to_pil(np_img: np.ndarray) -> Image.Image:
    '''this function converts a np array of jpeg encoded bytes to pil'''
    if np_img.dtype != np.uint8:
        raise TypeError(f"Expected np.ndarray of dtype np.uint8, but got type {np_img.dtype}")
    img_shape = np_img.shape
    if len(img_shape) != 1:
        raise ValueError(f"Expected np.ndarray with 1 dimensions, but got np.ndarray of shape {img_shape}")

    pil_img = Image.open(BytesIO(np_img.tobytes()))
    return pil_img


def np_rgb_to_pil(np_img: np.ndarray) -> Image.Image:
    if np_img.dtype != np.uint8:
        raise TypeError(f"Expected np.ndarray of dtype np.uint8, but got type {np_img.dtype}")
    img_shape = np_img.shape
    if len(img_shape) != 3:
        raise ValueError(f"Expected np.ndarray with 3 dimensions, but got np.ndarray of shape {img_shape}")
    if img_shape[-1] != 3:
        raise ValueError(f"Expected np.ndarray with 3 channels, but got np.ndarray of shape {img_shape}")

    pil_img = Image.fromarray(np_img, mode='RGB')
    return pil_img


def np_seg_to_pil(np_img: np.ndarray) -> Image.Image:
    if np_img.dtype != np.uint8:
        raise TypeError(f"Expected np.ndarray of dtype np.uint8, but got dtype {np_img.dtype}")
    if np_img.ndim == 3 and np_img.shape[-1] == 1:
        np_img = np.squeeze(np_img)
    elif np_img.ndim != 2:
        raise ValueError(f"Unsupported shape {np_img.shape}")

    pil_img = Image.fromarray(np_img, mode='L')
    return pil_img


def np_instance_to_pil(np_img: np.ndarray) -> Image.Image:
    if np_img.dtype != np.uint8:
        raise TypeError(f"Expected np.ndarray of dtype np.uint8, but got dtype {np_img.dtype}")
    if np_img.ndim == 3 and np_img.shape[-1] == 1:
        np_img = np.squeeze(np_img)
    elif np_img.ndim != 2:
        raise ValueError(f"Unsupported shape {np_img.shape}")

    pil_img = Image.fromarray(np_img, mode='L')
    return pil_img


def _check_inverse_depth_range(depth: np.ndarray):
    ignore_mask = depth == PAD_VALUE_DEPTH_FLOAT32
    low, high = np.min(depth[~ignore_mask]), np.max(depth[~ignore_mask])
    if low < 0.0 or high > 1.0:
        raise ValueError(f"Normalized depth must be in range [0, 1], but values are in range [{low}, {high}]")


def _check_depth_m_range(depth_m: np.ndarray):
    if depth_m.dtype != np.float32 and depth_m.dtype != np.float64:
        raise TypeError(f"Expected np.ndarray of dtype np.float32 or np.float64, but got dtype {depth_m.dtype}")

    low = np.min(depth_m[depth_m != PAD_VALUE_DEPTH_FLOAT32])
    high = np.max(depth_m[depth_m != PAD_VALUE_DEPTH_FLOAT32])
    if low < MIN_DEPTH_M:
        raise ValueError(f"Depth must be bigger than {MIN_DEPTH_M}, but values are in range [{low}, {high}]")


def _inverse_depth_normalized_uint16_to_float32(depth: np.ndarray) -> np.ndarray:
    """[0, 65534] -> [0, 1] and 65535 -> -1"""
    ignore_mask = depth == PAD_VALUE_DEPTH_UINT16
    depth = depth.astype('float32') / _DEPTH_NORM_CONST
    depth[ignore_mask] = PAD_VALUE_DEPTH_FLOAT32
    return depth


def _inverse_depth_normalized_float32_to_uint16(depth: np.ndarray) -> np.ndarray:
    """[0, 1] -> [0, 65534] and -1 -> 65535"""
    _check_inverse_depth_range(depth)
    ignore_mask = depth == PAD_VALUE_DEPTH_FLOAT32
    depth = depth * _DEPTH_NORM_CONST
    # If any values were 1, these get mapped to 65535, clip to 65534
    depth = np.clip(depth, 0, PAD_VALUE_DEPTH_UINT16 - 1)
    depth[ignore_mask] = PAD_VALUE_DEPTH_UINT16
    return depth.astype('uint16')


def np_inverse_depth_normalized_to_pil(np_img: np.ndarray) -> Image.Image:
    """
    Depth data is saved as inverse depth, represented as uint16, where a value of 0 corresponds to infinite
    depth in meters and a value of 2**16-1 corresponds to the minimum possible depth in meters (0.25m for Ningaloo)

    Ignore values of -1 get mapped to 65535 specially.
    """
    if np_img.dtype == np.float32:
        inverse_depth_unnormalized = _inverse_depth_normalized_float32_to_uint16(np_img)
    elif np_img.dtype == np.uint16:
        inverse_depth_unnormalized = np_img
    else:
        raise TypeError(f"Expected np.ndarray of dtype np.uint16 or np.float32, but got dtype {np_img.dtype}")

    if inverse_depth_unnormalized.ndim == 3 and inverse_depth_unnormalized.shape[-1] == 1:
        inverse_depth_unnormalized = np.squeeze(inverse_depth_unnormalized)
    elif inverse_depth_unnormalized.ndim != 2:
        raise ValueError(f"Unsupported shape {inverse_depth_unnormalized.shape}")

    # Hacky solution as PIL has a bug saving 16bit png
    pil_img = Image.new("I", inverse_depth_unnormalized.T.shape)
    pil_img.frombytes(inverse_depth_unnormalized.tobytes(), 'raw', "I;16")
    return pil_img


def np_inverse_depth_invm_to_pil(np_img: np.ndarray) -> Image.Image:
    inverse_depth = np_normalize_inverse_depth_invm(np_img)
    return np_inverse_depth_normalized_to_pil(inverse_depth)


def pil_inverse_depth_combine_channels(pil_img: Image.Image) -> np.ndarray:
    # PIL doesn't load correctly 16-bit grayscale image, even with .convert('I;16')
    inverse_depth_rg = np.asarray(pil_img, dtype=np.uint16)

    inverse_depth_unnormalized = inverse_depth_rg[:, :, 0] + inverse_depth_rg[:, :, 1] * 256
    return inverse_depth_unnormalized.reshape((pil_img.height, pil_img.width))


def pil_inverse_depth_to_np_uint16(pil_img: Image.Image) -> np.ndarray:
    """Converts a pil image (depth as a uint16 in range [0, 2^16-1]) to a 3d np array"""
    if pil_img.mode == 'I':  # depth images saved by Python
        inverse_depth_unnormalized = np.asarray(pil_img, dtype=np.uint16)
    elif pil_img.mode == 'LA':  # depth images saved by C++
        inverse_depth_unnormalized = pil_inverse_depth_combine_channels(pil_img)
    else:
        raise TypeError(f"Expected PIL image of mode 'I' or 'LA', but got mode {pil_img.mode}")

    return np.expand_dims(inverse_depth_unnormalized, axis=-1)


def pil_inverse_depth_to_np_float32(pil_img: Image.Image) -> np.ndarray:
    """Converts a pil image (depth as a uint16 in range [0, 2^16-1]) to a 3d np array in range [0, 1]"""
    inverse_depth_unnormalized = pil_inverse_depth_to_np_uint16(pil_img)
    inverse_depth_normalized = _inverse_depth_normalized_uint16_to_float32(inverse_depth_unnormalized)
    return inverse_depth_normalized


def pil_inverse_depth_to_invm(pil_img: Image.Image) -> np.ndarray:
    """Converts a pil image (depth as a uint16 in range [0, 2^16-1]) to a 3d np array with inverse depth in 1/metres
    2^16-1 is a special value which gets mapped to -1 or an ignore depth.
    """
    inverse_depth_normalized = pil_inverse_depth_to_np_float32(pil_img)
    inverse_depth_invm = np_inverse_depth_normalized_to_invm(inverse_depth_normalized)
    return inverse_depth_invm


def np_inverse_depth_normalized_to_invm(np_img: np.ndarray) -> np.ndarray:
    """Converts inverse depth either as a float [0, 1] or a uint16 [0, 2^16-1] to inverse depth in 1/metres"""
    # Ensure inverse depth is float32
    if np_img.dtype == np.uint16:
        inverse_depth_normalized = _inverse_depth_normalized_uint16_to_float32(np_img)
    elif np_img.dtype == np.float32:
        _check_inverse_depth_range(np_img)
        inverse_depth_normalized = np_img
    else:
        raise TypeError(f"Expected np.ndarray of dtype np.uint16 or np.float32, but got dtype {np_img.dtype}")

    invm = inverse_depth_normalized * MAX_INVDEPTH_INVM
    invm[inverse_depth_normalized == PAD_VALUE_DEPTH_FLOAT32] = PAD_VALUE_DEPTH_FLOAT32
    return invm


def np_normalize_inverse_depth_invm(invm: np.ndarray) -> np.ndarray:
    inverse_depth_normalized = np.clip(invm, 0, MAX_INVDEPTH_INVM) / MAX_INVDEPTH_INVM
    inverse_depth_normalized[invm == PAD_VALUE_DEPTH_FLOAT32] = PAD_VALUE_DEPTH_FLOAT32
    return inverse_depth_normalized


def np_inverse_depth_invm_to_depth_m(invm: np.ndarray) -> np.ndarray:
    # Ensure no division by 0
    depth_m = 1 / np.clip(invm, MIN_INVDEPTH_INVM, MAX_INVDEPTH_INVM)
    depth_m[invm == PAD_VALUE_DEPTH_FLOAT32] = PAD_VALUE_DEPTH_FLOAT32
    return depth_m


def np_depth_m_to_inverse_depth_invm(depth_m: np.ndarray) -> np.ndarray:
    invm = 1 / np.clip(depth_m, MIN_DEPTH_M, MAX_DEPTH_M)
    invm[depth_m == PAD_VALUE_DEPTH_FLOAT32] = PAD_VALUE_DEPTH_FLOAT32
    return invm

def np_inverse_depth_invm_to_depth_m_normalized(invm: np.ndarray) -> np.ndarray:
    # converts invese depth to meters normalized to range [-1, 1]
    # invm = np_inverse_depth_normalized_to_invm(invm)
    # invm = _inverse_depth_normalized_uint16_to_float32(invm)
    depth_m = np_inverse_depth_normalized_to_depth_m(invm)
    if not(depth_m.min() >= MIN_DEPTH_M and depth_m.max() <= MAX_DEPTH_M):
        depth_m = np.clip(depth_m, MIN_DEPTH_M, MAX_DEPTH_M)
    depth_m = (depth_m / MAX_DEPTH_M) * 2 - 1
    return depth_m


def np_inverse_depth_normalized_to_depth_m(np_img: np.ndarray) -> np.ndarray:
    """Converts inverse depth either as a float [0, 1] or a uint16 [0, 2^16-1] to depth in metres

    -1 gets mapped to -1
    """
    invm = np_inverse_depth_normalized_to_invm(np_img)
    return np_inverse_depth_invm_to_depth_m(invm)


def depth_m_to_inverse_depth_float32(depth_m: np.ndarray) -> np.ndarray:
    """Converts depth in metres to inverse depth in the range [0, 1]
    -1 gets mapped to -1
    """
    _check_depth_m_range(depth_m)
    invm = np_depth_m_to_inverse_depth_invm(depth_m)
    return np_normalize_inverse_depth_invm(invm)


def depth_m_to_inverse_depth_uint16(depth_m: np.ndarray) -> np.ndarray:
    """Converts depth in metres to inverse depth in the range [0, 2^16 - 1]"""
    _check_depth_m_range(depth_m)
    inverse_depth_normalized = depth_m_to_inverse_depth_float32(depth_m)
    return _inverse_depth_normalized_float32_to_uint16(inverse_depth_normalized)

# Normals

def normal_to_normalised_normal(normal):
    normal = normal / MAX_NORMAL
    return normal