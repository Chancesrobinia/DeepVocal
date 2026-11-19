"""Protect the saved model API key using the local Windows account."""

import ctypes
from ctypes import wintypes
import os


class CredentialError(Exception):
    pass


class DataBlob(ctypes.Structure):
    _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]


def transform(value: bytes, *, protect: bool) -> bytes:
    if os.name != "nt":
        raise CredentialError("此平台不支持保存 API Key，请使用服务端环境变量配置。")

    crypt32 = ctypes.WinDLL("Crypt32.dll", use_last_error=True)
    operation = crypt32.CryptProtectData if protect else crypt32.CryptUnprotectData
    blob_ptr = ctypes.POINTER(DataBlob)
    operation.argtypes = [blob_ptr, ctypes.c_void_p, blob_ptr, ctypes.c_void_p, ctypes.c_void_p,
                          wintypes.DWORD, blob_ptr]
    operation.restype = wintypes.BOOL
    free = ctypes.WinDLL("Kernel32.dll").LocalFree
    free.argtypes = [ctypes.c_void_p]
    free.restype = ctypes.c_void_p

    buffer = (ctypes.c_ubyte * len(value)).from_buffer_copy(value)
    source = DataBlob(len(value), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    target = DataBlob()
    try:
        if not operation(ctypes.byref(source), None, None, None, None, 0x1, ctypes.byref(target)):
            raise CredentialError("Windows 无法保护或读取该 API Key，请在当前账户下重新填写并保存。")
        return ctypes.string_at(target.data, target.size)
    finally:
        ctypes.memset(buffer, 0, len(value))
        if target.data:
            if not protect:
                ctypes.memset(target.data, 0, target.size)
            free(target.data)
