// $MLIRPATH/mlir-opt --mlir-print-debuginfo --convert-scf-to-cf --convert-to-llvm runtime.llvm.mlir | $MLIRPATH/mlir-translate --mlir-to-llvmir -o runtime.ll


module {
	// Declare printf
	llvm.func @printf(!llvm.ptr, ...) -> i32

	memref.global "private" constant @object$tag : memref<i64> = dense<0>
	memref.global "private" constant @int$tag : memref<i64> = dense<11>
	memref.global "private" constant @bool$tag : memref<i64> = dense<22>
	memref.global "private" constant @str$tag : memref<i64> = dense<26>
	memref.global "private" constant @float$tag : memref<i64> = dense<10>

	llvm.mlir.global @print$list_begin("[\00" : i8) : !llvm.array<2 x i8>
	llvm.mlir.global @print$list_comma(", \00" : i8) : !llvm.array<3 x i8>
	llvm.mlir.global @print$list_end("]\00" : i8) : !llvm.array<2 x i8>
	llvm.mlir.global @print$none("None\n\00" : i8) : !llvm.array<6 x i8>
	llvm.mlir.global @print$object_format("<%p>\n\00" : i8) : !llvm.array<6 x i8>
	llvm.mlir.global @print$str_format("%s\n\00" : i8) : !llvm.array<4 x i8>
	llvm.mlir.global @print$int_format("%ld\n\00" : i8) : !llvm.array<5 x i8>
	llvm.mlir.global @print$bool_true("True\n\00" : i8) : !llvm.array<6 x i8>
	llvm.mlir.global @print$bool_false("False\n\00" : i8) : !llvm.array<7 x i8>
	llvm.mlir.global @print$float_format("%lg\n\00" : i8) : !llvm.array<5 x i8>

	func.func public @allocate$i64s(%count: index) -> memref<?xi8> {
		%sizeof_i64 = arith.constant 8 : index
		%size = arith.muli %count, %sizeof_i64 : index

		%memref = memref.alloc(%size) : memref<?xi8>
		func.return %memref : memref<?xi8>
	}

	func.func public @as$i64s(%memref: memref<?xi8>) -> memref<?xi64> {
		%sizeof_i64 = arith.constant 8 : index
		%2, %3, %size_bytes, %4 = memref.extract_strided_metadata %memref : memref<?xi8> -> memref<i8>, index, index, index
		%size = arith.divsi %size_bytes, %sizeof_i64 : index

		%offset = arith.constant 0: index
		%out = memref.view %memref[%offset][%size] : memref<?xi8> to memref<?xi64>
		func.return %out : memref<?xi64>
	}

	func.func public @as$memrefs(%memref: memref<?xi8>, %offset_bytes: index) -> memref<?xmemref<?xi8>> {
		// %sizeof_memref = arith.constant 5 * sizeof(i64) : index
		%sizeof_memref = arith.constant 40 : index
		%2, %3, %size_bytes, %4 = memref.extract_strided_metadata %memref : memref<?xi8> -> memref<i8>, index, index, index
		%size = arith.divsi %size_bytes, %sizeof_memref : index

		%out = memref.view %memref[%offset_bytes][%size] : memref<?xi8> to memref<?xmemref<?xi8>>
		func.return %out : memref<?xmemref<?xi8>>
	}

	func.func public @assert$size(%memref: memref<?xi8>, %size : index) {
		%2, %3, %actual, %4 = memref.extract_strided_metadata %memref : memref<?xi8> -> memref<i8>, index, index, index
		%eq = arith.cmpi eq, %size, %actual : index
		cf.assert %eq, "Object size assertion failed"
		func.return
	}

	func.func public @assert$i64_count(%memref: memref<?xi8>, %count : index) {
		%sizeof_i64 = arith.constant 8 : index
		%expected = arith.muli %count, %sizeof_i64 : index
		func.call @assert$size(%memref, %expected) : (memref<?xi8>, index) -> ()
		func.return
	}

	func.func public @__tag__$object(%memref : memref<?xi8>) -> i64 {
		%zero = arith.constant 0 : index
		%i64s = func.call @as$i64s(%memref) : (memref<?xi8>) -> memref<?xi64>
		%tag = memref.load %i64s[%zero] : memref<?xi64>	
		func.return %tag : i64
	}

	func.func public @__print__$object(%memref : memref<?xi8>) -> memref<?xi8> {
		%i = memref.extract_aligned_pointer_as_index %memref : memref<?xi8> -> index
		%p = arith.index_cast %i : index to i64
		%f = llvm.mlir.addressof @print$object_format : !llvm.ptr
		%0 = llvm.call @printf(%f, %p) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
		%1 = arith.extsi %0 : i32 to i64
		%2 = func.call @__box__$int(%1) : (i64) -> memref<?xi8>
		func.return %2 : memref<?xi8>
	}

	func.func public @__assert__$int(%memref: memref<?xi8>) {
		%0 = memref.get_global @int$tag : memref<i64>
		%expected_tag = memref.load %0[] : memref<i64>
		%actual_tag = func.call @__tag__$object(%memref) : (memref<?xi8>) -> i64
		%eq = arith.cmpi eq, %expected_tag, %actual_tag : i64
		cf.assert %eq, "Provided object is not an int"

		%two = arith.constant 2 : index
		func.call @assert$i64_count(%memref, %two) : (memref<?xi8>, index) -> ()
		func.return
	}

	func.func public @__box__$int(%v: i64) -> memref<?xi8> {
		%0 = memref.get_global @int$tag : memref<i64>
		%tag = memref.load %0[] : memref<i64>

		// %size = arith.constant 2 * sizeof(i64) : index
		%count = arith.constant 2 : index
		%out = func.call @allocate$i64s(%count) : (index) -> memref<?xi8>
		%out_i64s = func.call @as$i64s(%out) : (memref<?xi8>) -> memref<?xi64>

		%zero = arith.constant 0 : index
		memref.store %tag, %out_i64s[%zero] : memref<?xi64>
		%one = arith.constant 1 : index
		memref.store %v, %out_i64s[%one] : memref<?xi64>

		func.return %out : memref<?xi8>
	}

	func.func public @__unbox__$int(%memref: memref<?xi8>) -> i64 {
		func.call @__assert__$int(%memref) : (memref<?xi8>) -> ()
		%i64s = func.call @as$i64s(%memref) : (memref<?xi8>) -> memref<?xi64>

		%one = arith.constant 1 : index
		%i = memref.load %i64s[%one] : memref<?xi64>
		func.return %i : i64
	}

	func.func public @__print__$int(%memref : memref<?xi8>) -> memref<?xi8> {
		%i = func.call @__unbox__$int(%memref) : (memref<?xi8>) -> i64
		%f = llvm.mlir.addressof @print$int_format : !llvm.ptr
		%0 = llvm.call @printf(%f, %i) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i64) -> i32
		%1 = arith.extsi %0 : i32 to i64
		%2 = func.call @__box__$int(%1) : (i64) -> memref<?xi8>
		func.return %2 : memref<?xi8>
	}



	func.func public @__assert__$float(%memref: memref<?xi8>) {
		%0 = memref.get_global @float$tag : memref<i64>
		%expected_tag = memref.load %0[] : memref<i64>
		%actual_tag = func.call @__tag__$object(%memref) : (memref<?xi8>) -> i64
		%eq = arith.cmpi eq, %expected_tag, %actual_tag : i64
		cf.assert %eq, "Provided object is not an int"

		%two = arith.constant 2 : index
		func.call @assert$i64_count(%memref, %two) : (memref<?xi8>, index) -> ()
		func.return
	}

	func.func public @__box__$float(%v: f64) -> memref<?xi8> {
		%0 = memref.get_global @float$tag : memref<i64>
		%tag = memref.load %0[] : memref<i64>

		// %size = arith.constant 2 * sizeof(i64) : index
		%count = arith.constant 2 : index
		%out = func.call @allocate$i64s(%count) : (index) -> memref<?xi8>
		%out_i64s = func.call @as$i64s(%out) : (memref<?xi8>) -> memref<?xi64>

		%i = arith.bitcast %v : f64 to i64
		%zero = arith.constant 0 : index
		memref.store %tag, %out_i64s[%zero] : memref<?xi64>
		%one = arith.constant 1 : index
		memref.store %i, %out_i64s[%one] : memref<?xi64>

		func.return %out : memref<?xi8>
	}

	func.func public @__unbox__$float(%memref: memref<?xi8>) -> f64 {
		func.call @__assert__$float(%memref) : (memref<?xi8>) -> ()
		%i64s = func.call @as$i64s(%memref) : (memref<?xi8>) -> memref<?xi64>

		%one = arith.constant 1 : index
		%i = memref.load %i64s[%one] : memref<?xi64>
		%f = arith.bitcast %i : i64 to f64
		func.return %f : f64
	}

	func.func public @__print__$float(%memref : memref<?xi8>) -> memref<?xi8> {
		%v = func.call @__unbox__$float(%memref) : (memref<?xi8>) -> f64
		%f = llvm.mlir.addressof @print$float_format : !llvm.ptr
		%0 = llvm.call @printf(%f, %v) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
		%1 = arith.extsi %0 : i32 to i64
		%2 = func.call @__box__$int(%1) : (i64) -> memref<?xi8>
		func.return %2 : memref<?xi8>
	}



	func.func public @__print__$str(%memref : memref<?xi8>) -> memref<?xi8> {
		// func.call @__assert__$srt(%memref) : (memref<?xi8>) -> ()
		%0 = memref.extract_aligned_pointer_as_index %memref : memref<?xi8> -> index
		// %offset = arith.constant 2 * sizeof(i64) : index
		%offset = arith.constant 16 : index
		%1 = arith.addi %0, %offset : index
		%2 = arith.index_cast %1 : index to i64
		%p = llvm.inttoptr %2 : i64 to !llvm.ptr

		%f = llvm.mlir.addressof @print$str_format : !llvm.ptr
		%3 = llvm.call @printf(%f, %p) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
		%4 = arith.extsi %3 : i32 to i64
		%5 = func.call @__box__$int(%4) : (i64) -> memref<?xi8>
		func.return %5 : memref<?xi8>
	}

	func.func private @__print__$object$dispatcher(%memref : memref<?xi8>) -> memref<?xi8>
	func.func @print(%memref : memref<?xi8>) -> memref<?xi8> {
		%0 = func.call @__print__$object$dispatcher(%memref) : (memref<?xi8>) -> memref<?xi8>
		func.return %0 : memref<?xi8>
	}
}
