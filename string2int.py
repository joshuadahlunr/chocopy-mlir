def string_to_i64_hex(s: str):
    data = s.encode("utf-8")

    # Pad to a multiple of 8 bytes
    if len(data) % 8 != 0:
        data += b"\x00" * (8 - (len(data) % 8))

    i64s = []
    for i in range(0, len(data), 8):
        chunk = data[i:i+8]
        value = int.from_bytes(chunk, byteorder="little", signed=True)
        i64s.append(value)

    return [f"0x{value & 0xffffffffffffffff:016x}" for value in i64s]


if __name__ == "__main__":
    s = "Hello from MLIR!\0"
    for v in string_to_i64_hex(s):
        print(v)
    print(len(s))