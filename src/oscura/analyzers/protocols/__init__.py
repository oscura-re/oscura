"""Protocol decoder package.

Provides protocol decoders for common serial and automotive protocols including
UART, SPI, I2C, CAN, LIN, FlexRay, JTAG, SWD, I2S, USB, HDLC, Manchester, CAN-FD,
and 1-Wire.
"""

from oscura.analyzers.protocols.base import (
    Annotation,
    AnnotationLevel,
    AsyncDecoder,
    ChannelDef,
    DecoderState,
    OptionDef,
    ProtocolDecoder,
    SyncDecoder,
)
from oscura.analyzers.protocols.can import (
    CAN_BITRATES,
    CANDecoder,
    CANFrame,
    CANFrameType,
    decode_can,
)
from oscura.analyzers.protocols.can_fd import (
    CANFD_DLC_TO_LENGTH,
    CANFDDecoder,
    CANFDFrame,
    CANFDFrameType,
    decode_can_fd,
)
from oscura.analyzers.protocols.flexray import (
    FlexRayDecoder,
    FlexRayFrame,
    FlexRaySegment,
    decode_flexray,
)
from oscura.analyzers.protocols.hdlc import HDLCDecoder, decode_hdlc
from oscura.analyzers.protocols.i2c import I2CDecoder, decode_i2c
from oscura.analyzers.protocols.i2s import I2SDecoder, I2SMode, decode_i2s
from oscura.analyzers.protocols.jtag import (
    JTAG_INSTRUCTIONS,
    JTAGDecoder,
    TAPState,
    decode_jtag,
)
from oscura.analyzers.protocols.lin import LINDecoder, LINVersion, decode_lin
from oscura.analyzers.protocols.manchester import (
    ManchesterDecoder,
    ManchesterMode,
    decode_manchester,
)
from oscura.analyzers.protocols.onewire import (
    FAMILY_CODES,
    ROM_COMMAND_NAMES,
    OneWireDecoder,
    OneWireMode,
    OneWireROMCommand,
    OneWireROMID,
    OneWireTimings,
    decode_onewire,
)
from oscura.analyzers.protocols.spi import SPIDecoder, decode_spi
from oscura.analyzers.protocols.swd import SWDDecoder, SWDResponse, decode_swd
from oscura.analyzers.protocols.uart import UARTDecoder, decode_uart
from oscura.analyzers.protocols.usb import (
    PID_NAMES,
    USBPID,
    USBDecoder,
    USBSpeed,
    decode_usb,
)

__all__ = [
    "CANFD_DLC_TO_LENGTH",
    "CAN_BITRATES",
    # 1-Wire (PRO-007)
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
    # 1-Wire
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
