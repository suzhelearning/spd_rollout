"""Private socketpair framing for the policy process and its Isaac Sim child."""
import json
import struct


def receive_exact(channel, count):
    data = bytearray(count)
    view = memoryview(data)
    offset = 0
    while offset < count:
        received = channel.recv_into(view[offset:])
        if not received:
            raise EOFError("Isaac Sim render channel closed")
        offset += received
    return data


def send_packet(channel, metadata, payload=b""):
    header = json.dumps({**metadata, "payload_bytes": len(payload)}, allow_nan=False).encode()
    channel.sendall(struct.pack("!I", len(header)))
    channel.sendall(header)
    if payload:
        channel.sendall(payload)


def receive_packet(channel):
    length = struct.unpack("!I", receive_exact(channel, 4))[0]
    if not 0 < length <= 1_048_576:
        raise ValueError("Invalid renderer message header length")
    metadata = json.loads(receive_exact(channel, length))
    size = metadata.pop("payload_bytes")
    if type(size) is not int or not 0 <= size <= 16_777_216:
        raise ValueError("Invalid renderer image length")
    return metadata, receive_exact(channel, size)
