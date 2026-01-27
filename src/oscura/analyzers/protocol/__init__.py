"""Protocol decoding module.

.. deprecated:: 0.6.0
    This module (singular 'protocol') is deprecated in favor of 'protocols' (plural).
    Use ``from oscura.analyzers.protocols import UARTDecoder`` instead.
    This module will be removed in v1.0.0.

This module re-exports protocol decoders from the protocols package
for convenient access. Both import paths are equivalent:

    from oscura.analyzers.protocol import UARTDecoder  # singular (deprecated)
    from oscura.analyzers.protocols import UARTDecoder  # plural (recommended)

The plural form (protocols) is recommended as the canonical import path.
See IMPORT-PATHS.md in the repository root for detailed guidelines.
"""

import warnings

# Issue deprecation warning when this module is imported
warnings.warn(
    "Importing from 'oscura.analyzers.protocol' (singular) is deprecated. "
    "Use 'oscura.analyzers.protocols' (plural) instead. "
    "This module will be removed in v1.0.0.",
    DeprecationWarning,
    stacklevel=2,
)

from oscura.analyzers.protocols import (
    CAN_BITRATES,
    CANFD_DLC_TO_LENGTH,
    FAMILY_CODES,
    JTAG_INSTRUCTIONS,
    PID_NAMES,
    ROM_COMMAND_NAMES,
    USBPID,
    Annotation,
    AnnotationLevel,
    AsyncDecoder,
    CANDecoder,
    CANFDDecoder,
    CANFDFrame,
    CANFDFrameType,
    CANFrame,
    CANFrameType,
    ChannelDef,
    DecoderState,
    FlexRayDecoder,
    FlexRayFrame,
    FlexRaySegment,
    HDLCDecoder,
    I2CDecoder,
    I2SDecoder,
    I2SMode,
    JTAGDecoder,
    LINDecoder,
    LINVersion,
    ManchesterDecoder,
    ManchesterMode,
    OneWireDecoder,
    OneWireMode,
    OneWireROMCommand,
    OneWireROMID,
    OneWireTimings,
    OptionDef,
    ProtocolDecoder,
    SPIDecoder,
    SWDDecoder,
    SWDResponse,
    SyncDecoder,
    TAPState,
    UARTDecoder,
    USBDecoder,
    USBSpeed,
    decode_can,
    decode_can_fd,
    decode_flexray,
    decode_hdlc,
    decode_i2c,
    decode_i2s,
    decode_jtag,
    decode_lin,
    decode_manchester,
    decode_onewire,
    decode_spi,
    decode_swd,
    decode_uart,
    decode_usb,
)

__all__ = [
    "CANFD_DLC_TO_LENGTH",
    "CAN_BITRATES",
    "FAMILY_CODES",
    "JTAG_INSTRUCTIONS",
    "PID_NAMES",
    "ROM_COMMAND_NAMES",
    "USBPID",
    "Annotation",
    "AnnotationLevel",
    "AsyncDecoder",
    # CAN (PRO-005)
    "CANDecoder",
    # CAN-FD (PRO-015)
    "CANFDDecoder",
    "CANFDFrame",
    "CANFDFrameType",
    "CANFrame",
    "CANFrameType",
    "ChannelDef",
    "DecoderState",
    # FlexRay (PRO-016)
    "FlexRayDecoder",
    "FlexRayFrame",
    "FlexRaySegment",
    # HDLC (PRO-013)
    "HDLCDecoder",
    # I2C (PRO-004)
    "I2CDecoder",
    # I2S (PRO-011)
    "I2SDecoder",
    "I2SMode",
    # JTAG (PRO-009)
    "JTAGDecoder",
    # LIN (PRO-008)
    "LINDecoder",
    "LINVersion",
    # Manchester (PRO-014)
    "ManchesterDecoder",
    "ManchesterMode",
    # 1-Wire (PRO-007)
    "OneWireDecoder",
    "OneWireMode",
    "OneWireROMCommand",
    "OneWireROMID",
    "OneWireTimings",
    "OptionDef",
    # Base
    "ProtocolDecoder",
    # SPI (PRO-003)
    "SPIDecoder",
    # SWD (PRO-010)
    "SWDDecoder",
    "SWDResponse",
    "SyncDecoder",
    "TAPState",
    # UART (PRO-002)
    "UARTDecoder",
    # USB (PRO-012)
    "USBDecoder",
    "USBSpeed",
    "decode_can",
    "decode_can_fd",
    "decode_flexray",
    "decode_hdlc",
    "decode_i2c",
    "decode_i2s",
    "decode_jtag",
    "decode_lin",
    "decode_manchester",
    "decode_onewire",
    "decode_spi",
    "decode_swd",
    "decode_uart",
    "decode_usb",
]
