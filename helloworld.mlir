// $MLIRPATH/mlir-opt --convert-scf-to-cf --convert-to-llvm helloworld.mlir | $MLIRPATH/mlir-translate --mlir-to-llvmir
// $MLIRPATH/mlir-opt -verify-diagnostics --convert-scf-to-cf --convert-to-llvm build/gen.mlir | $MLIRPATH/mlir-translate --mlir-to-llvmir

module {
	func.func private @allocate$i64s(%count: index) -> memref<?xi8>
	func.func private @as$i64s(%memref: memref<?xi8>) -> memref<?xi64>
	func.func private @as$memrefs(%memref: memref<?xi8>, %offset_bytes: index) -> memref<?xmemref<?xi8>>
	func.func private @assert$size(%memref: memref<?xi8>, %size : index)
	func.func private @assert$i64_count(%memref: memref<?xi8>, %count : index)
	func.func private @__tag__$object(%memref : memref<?xi8>) -> i64
	func.func private @__print__$object(%memref : memref<?xi8>) -> memref<?xi8>
	func.func private @__assert__$int(%memref: memref<?xi8>)
	func.func private @__box__$int(%v: i64) -> memref<?xi8>
	func.func private @__unbox__$int(%memref: memref<?xi8>) -> i64
	func.func private @__print__$int(%memref : memref<?xi8>) -> memref<?xi8>
	func.func private @__print__$float(%memref : memref<?xi8>) -> memref<?xi8>
	func.func private @__print__$str(%memref : memref<?xi8>) -> memref<?xi8>

	func.func private @print(%memref : memref<?xi8>) -> memref<?xi8>

	memref.global "private" constant @hello_string : memref<40xi8> = dense<[26, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x48, 0x65, 0x6c, 0x6c, 0x6f, 0x20, 0x66, 0x72,
		0x6f, 0x6d, 0x20, 0x4d, 0x4c, 0x49, 0x52, 0x21,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
	]>
	memref.global "private" constant @other_string : memref<40xi8> = dense<[26, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x54, 0x68, 0x69, 0x73, 0x20, 0x69, 0x73, 0x20, 
		0x61, 0x6e, 0x6f, 0x74, 0x68, 0x65, 0x72, 0x20, 
		0x73, 0x74, 0x72, 0x69, 0x6e, 0x67, 0x00, 0x00
	]>

	func.func @__print__$object$dispatcher(%memref : memref<?xi8>) -> memref<?xi8> {
		%0 = func.call @__tag__$object(%memref) : (memref<?xi8>) -> i64
		%tag = arith.index_cast %0 : i64 to index
		%1 = scf.index_switch %tag -> memref<?xi8>
		case 11 { // int
			%2 = func.call @__print__$int(%memref) : (memref<?xi8>) -> memref<?xi8>
			scf.yield %2 : memref<?xi8>
		}
		case 26 { // str
			%2 = func.call @__print__$str(%memref) : (memref<?xi8>) -> memref<?xi8>
			scf.yield %2 : memref<?xi8>
		}
		default {
			// %2 = func.call @__print__$object(%memref) : (memref<?xi8>) -> memref<?xi8>
			// scf.yield %2 : memref<?xi8>
			%false = arith.constant 0 : i1
			cf.assert %false, "Type does not support __print__"
			%one = arith.constant 1 : index
			%2 = memref.alloc(%one) : memref<?xi8>
			scf.yield %2 : memref<?xi8>
		}
		func.return %1 : memref<?xi8>
	}

	func.func @main() -> i32 {
		// Get a reference to the string buffer
		%0 = memref.get_global @hello_string : memref<40xi8>
		%1 = memref.cast %0 : memref<40xi8> to memref<?xi8>
		%2 = memref.get_global @other_string : memref<40xi8>
		%3 = memref.cast %2 : memref<40xi8> to memref<?xi8>

		// Call print
		%4 = func.call @print(%1) : (memref<?xi8>) -> memref<?xi8>
		%5 = func.call @print(%3) : (memref<?xi8>) -> memref<?xi8>
		// Unbox its return
		%6 = func.call @__unbox__$int(%4) : (memref<?xi8>) -> i64
		%7 = arith.trunci %6 : i64 to i32
		return %7 : i32
	}
}
