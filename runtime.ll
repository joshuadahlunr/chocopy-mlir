; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@assert_msg_0 = private constant [30 x i8] c"Provided object is not an int\00"
@assert_msg = private constant [29 x i8] c"Object size assertion failed\00"
@"object$tag" = private constant i64 0
@"int$tag" = private constant i64 11
@"bool$tag" = private constant i64 22
@"str$tag" = private constant i64 26
@"float$tag" = private constant i64 10
@"print$list_begin" = global [2 x i8] c"[\00"
@"print$list_comma" = global [3 x i8] c", \00"
@"print$list_end" = global [2 x i8] c"]\00"
@"print$none" = global [6 x i8] c"None\0A\00"
@"print$object_format" = global [6 x i8] c"<%p>\0A\00"
@"print$str_format" = global [4 x i8] c"%s\0A\00"
@"print$int_format" = global [5 x i8] c"%ld\0A\00"
@"print$bool_true" = global [6 x i8] c"True\0A\00"
@"print$bool_false" = global [7 x i8] c"False\0A\00"
@"print$float_format" = global [5 x i8] c"%lg\0A\00"

declare void @abort()

declare void @puts(ptr)

declare ptr @malloc(i64)

declare i32 @printf(ptr, ...)

define { ptr, ptr, i64, [1 x i64], [1 x i64] } @"allocate$i64s"(i64 %0) {
  %2 = mul i64 %0, 8
  %3 = getelementptr i8, ptr null, i64 %2
  %4 = ptrtoint ptr %3 to i64
  %5 = call ptr @malloc(i64 %4)
  %6 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %5, 0
  %7 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, ptr %5, 1
  %8 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, i64 0, 2
  %9 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %8, i64 %2, 3, 0
  %10 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %9, i64 1, 4, 0
  ret { ptr, ptr, i64, [1 x i64], [1 x i64] } %10
}

define { ptr, ptr, i64, [1 x i64], [1 x i64] } @"as$i64s"(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4) {
  %6 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %0, 0
  %7 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, ptr %1, 1
  %8 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, i64 %2, 2
  %9 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %8, i64 %3, 3, 0
  %10 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %9, i64 %4, 4, 0
  %11 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 0
  %12 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 1
  %13 = insertvalue { ptr, ptr, i64 } poison, ptr %11, 0
  %14 = insertvalue { ptr, ptr, i64 } %13, ptr %12, 1
  %15 = insertvalue { ptr, ptr, i64 } %14, i64 0, 2
  %16 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 2
  %17 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 3, 0
  %18 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 4, 0
  %19 = sdiv i64 %17, 8
  %20 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 0
  %21 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %20, 0
  %22 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 1
  %23 = getelementptr i8, ptr %22, i64 0
  %24 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %21, ptr %23, 1
  %25 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %24, i64 0, 2
  %26 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %25, i64 %19, 3, 0
  %27 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %26, i64 1, 4, 0
  ret { ptr, ptr, i64, [1 x i64], [1 x i64] } %27
}

define { ptr, ptr, i64, [1 x i64], [1 x i64] } @"as$memrefs"(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5) {
  %7 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %0, 0
  %8 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, ptr %1, 1
  %9 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %8, i64 %2, 2
  %10 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %9, i64 %3, 3, 0
  %11 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, i64 %4, 4, 0
  %12 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 0
  %13 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 1
  %14 = insertvalue { ptr, ptr, i64 } poison, ptr %12, 0
  %15 = insertvalue { ptr, ptr, i64 } %14, ptr %13, 1
  %16 = insertvalue { ptr, ptr, i64 } %15, i64 0, 2
  %17 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 2
  %18 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 3, 0
  %19 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 4, 0
  %20 = sdiv i64 %18, 40
  %21 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 0
  %22 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %21, 0
  %23 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 1
  %24 = getelementptr i8, ptr %23, i64 %5
  %25 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %22, ptr %24, 1
  %26 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %25, i64 0, 2
  %27 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %26, i64 %20, 3, 0
  %28 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %27, i64 1, 4, 0
  ret { ptr, ptr, i64, [1 x i64], [1 x i64] } %28
}

define void @"assert$size"(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5) {
  %7 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %0, 0
  %8 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, ptr %1, 1
  %9 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %8, i64 %2, 2
  %10 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %9, i64 %3, 3, 0
  %11 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, i64 %4, 4, 0
  %12 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 0
  %13 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 1
  %14 = insertvalue { ptr, ptr, i64 } poison, ptr %12, 0
  %15 = insertvalue { ptr, ptr, i64 } %14, ptr %13, 1
  %16 = insertvalue { ptr, ptr, i64 } %15, i64 0, 2
  %17 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 2
  %18 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 3, 0
  %19 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 4, 0
  %20 = icmp eq i64 %5, %18
  br i1 %20, label %21, label %22

21:                                               ; preds = %6
  ret void

22:                                               ; preds = %6
  call void @puts(ptr @assert_msg)
  call void @abort()
  unreachable
}

define void @"assert$i64_count"(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5) {
  %7 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %0, 0
  %8 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, ptr %1, 1
  %9 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %8, i64 %2, 2
  %10 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %9, i64 %3, 3, 0
  %11 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, i64 %4, 4, 0
  %12 = mul i64 %5, 8
  %13 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 0
  %14 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 1
  %15 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 2
  %16 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 3, 0
  %17 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 4, 0
  call void @"assert$size"(ptr %13, ptr %14, i64 %15, i64 %16, i64 %17, i64 %12)
  ret void
}

define i64 @"__tag__$object"(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4) {
  %6 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %0, 0
  %7 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, ptr %1, 1
  %8 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, i64 %2, 2
  %9 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %8, i64 %3, 3, 0
  %10 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %9, i64 %4, 4, 0
  %11 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 0
  %12 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 1
  %13 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 2
  %14 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 3, 0
  %15 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 4, 0
  %16 = call { ptr, ptr, i64, [1 x i64], [1 x i64] } @"as$i64s"(ptr %11, ptr %12, i64 %13, i64 %14, i64 %15)
  %17 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, 1
  %18 = getelementptr inbounds nuw i64, ptr %17, i64 0
  %19 = load i64, ptr %18, align 4
  ret i64 %19
}

define { ptr, ptr, i64, [1 x i64], [1 x i64] } @"__print__$object"(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4) {
  %6 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %0, 0
  %7 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, ptr %1, 1
  %8 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, i64 %2, 2
  %9 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %8, i64 %3, 3, 0
  %10 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %9, i64 %4, 4, 0
  %11 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 1
  %12 = ptrtoint ptr %11 to i64
  %13 = call i32 (ptr, ...) @printf(ptr @"print$object_format", i64 %12)
  %14 = sext i32 %13 to i64
  %15 = call { ptr, ptr, i64, [1 x i64], [1 x i64] } @"__box__$int"(i64 %14)
  ret { ptr, ptr, i64, [1 x i64], [1 x i64] } %15
}

define void @"__assert__$int"(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4) {
  %6 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %0, 0
  %7 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, ptr %1, 1
  %8 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, i64 %2, 2
  %9 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %8, i64 %3, 3, 0
  %10 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %9, i64 %4, 4, 0
  %11 = load i64, ptr @"int$tag", align 4
  %12 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 0
  %13 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 1
  %14 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 2
  %15 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 3, 0
  %16 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 4, 0
  %17 = call i64 @"__tag__$object"(ptr %12, ptr %13, i64 %14, i64 %15, i64 %16)
  %18 = icmp eq i64 %11, %17
  br i1 %18, label %19, label %25

19:                                               ; preds = %5
  %20 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 0
  %21 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 1
  %22 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 2
  %23 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 3, 0
  %24 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 4, 0
  call void @"assert$i64_count"(ptr %20, ptr %21, i64 %22, i64 %23, i64 %24, i64 2)
  ret void

25:                                               ; preds = %5
  call void @puts(ptr @assert_msg_0)
  call void @abort()
  unreachable
}

define { ptr, ptr, i64, [1 x i64], [1 x i64] } @"__box__$int"(i64 %0) {
  %2 = load i64, ptr @"int$tag", align 4
  %3 = call { ptr, ptr, i64, [1 x i64], [1 x i64] } @"allocate$i64s"(i64 2)
  %4 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %3, 0
  %5 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %3, 1
  %6 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %3, 2
  %7 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %3, 3, 0
  %8 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %3, 4, 0
  %9 = call { ptr, ptr, i64, [1 x i64], [1 x i64] } @"as$i64s"(ptr %4, ptr %5, i64 %6, i64 %7, i64 %8)
  %10 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %9, 1
  %11 = getelementptr inbounds nuw i64, ptr %10, i64 0
  store i64 %2, ptr %11, align 4
  %12 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %9, 1
  %13 = getelementptr inbounds nuw i64, ptr %12, i64 1
  store i64 %0, ptr %13, align 4
  ret { ptr, ptr, i64, [1 x i64], [1 x i64] } %3
}

define i64 @"__unbox__$int"(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4) {
  %6 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %0, 0
  %7 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, ptr %1, 1
  %8 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, i64 %2, 2
  %9 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %8, i64 %3, 3, 0
  %10 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %9, i64 %4, 4, 0
  %11 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 0
  %12 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 1
  %13 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 2
  %14 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 3, 0
  %15 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 4, 0
  call void @"__assert__$int"(ptr %11, ptr %12, i64 %13, i64 %14, i64 %15)
  %16 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 0
  %17 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 1
  %18 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 2
  %19 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 3, 0
  %20 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 4, 0
  %21 = call { ptr, ptr, i64, [1 x i64], [1 x i64] } @"as$i64s"(ptr %16, ptr %17, i64 %18, i64 %19, i64 %20)
  %22 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %21, 1
  %23 = getelementptr inbounds nuw i64, ptr %22, i64 1
  %24 = load i64, ptr %23, align 4
  ret i64 %24
}

define { ptr, ptr, i64, [1 x i64], [1 x i64] } @"__print__$int"(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4) {
  %6 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %0, 0
  %7 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, ptr %1, 1
  %8 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, i64 %2, 2
  %9 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %8, i64 %3, 3, 0
  %10 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %9, i64 %4, 4, 0
  %11 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 0
  %12 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 1
  %13 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 2
  %14 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 3, 0
  %15 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 4, 0
  %16 = call i64 @"__unbox__$int"(ptr %11, ptr %12, i64 %13, i64 %14, i64 %15)
  %17 = call i32 (ptr, ...) @printf(ptr @"print$int_format", i64 %16)
  %18 = sext i32 %17 to i64
  %19 = call { ptr, ptr, i64, [1 x i64], [1 x i64] } @"__box__$int"(i64 %18)
  ret { ptr, ptr, i64, [1 x i64], [1 x i64] } %19
}

define { ptr, ptr, i64, [1 x i64], [1 x i64] } @"__print__$str"(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4) {
  %6 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %0, 0
  %7 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, ptr %1, 1
  %8 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, i64 %2, 2
  %9 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %8, i64 %3, 3, 0
  %10 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %9, i64 %4, 4, 0
  %11 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 1
  %12 = ptrtoint ptr %11 to i64
  %13 = add i64 %12, 16
  %14 = inttoptr i64 %13 to ptr
  %15 = call i32 (ptr, ...) @printf(ptr @"print$str_format", ptr %14)
  %16 = sext i32 %15 to i64
  %17 = call { ptr, ptr, i64, [1 x i64], [1 x i64] } @"__box__$int"(i64 %16)
  ret { ptr, ptr, i64, [1 x i64], [1 x i64] } %17
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
