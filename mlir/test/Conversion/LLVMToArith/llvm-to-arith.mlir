// RUN: mlir-opt -convert-llvm-to-arith %s | FileCheck %s

// CHECK-LABEL: llvm.func @add(
// CHECK-SAME:    %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32
// CHECK:         %[[RES:.*]] = arith.addi %[[ARG0]], %[[ARG1]] : i32
// CHECK:         llvm.return %[[RES]] : i32
llvm.func @add(%lhs: i32, %rhs: i32) -> i32 {
  %0 = llvm.add %lhs, %rhs : i32
  llvm.return %0 : i32
}

// CHECK-LABEL: llvm.func @add_overflow_flags(
// CHECK:         %[[RES:.*]] = arith.addi %{{.*}}, %{{.*}} overflow<nsw, nuw> : i32
llvm.func @add_overflow_flags(%lhs: i32, %rhs: i32) -> i32 {
  %0 = llvm.add %lhs, %rhs overflow<nsw, nuw> : i32
  llvm.return %0 : i32
}

// CHECK-LABEL: llvm.func @sub(
// CHECK:         %[[RES:.*]] = arith.subi %{{.*}}, %{{.*}} : i64
llvm.func @sub(%lhs: i64, %rhs: i64) -> i64 {
  %0 = llvm.sub %lhs, %rhs : i64
  llvm.return %0 : i64
}

// CHECK-LABEL: llvm.func @sub_overflow_flags(
// CHECK:         %[[RES:.*]] = arith.subi %{{.*}}, %{{.*}} overflow<nsw> : i64
llvm.func @sub_overflow_flags(%lhs: i64, %rhs: i64) -> i64 {
  %0 = llvm.sub %lhs, %rhs overflow<nsw> : i64
  llvm.return %0 : i64
}

// CHECK-LABEL: llvm.func @mul(
// CHECK:         %[[RES:.*]] = arith.muli %{{.*}}, %{{.*}} : i32
llvm.func @mul(%lhs: i32, %rhs: i32) -> i32 {
  %0 = llvm.mul %lhs, %rhs : i32
  llvm.return %0 : i32
}

// CHECK-LABEL: llvm.func @sdiv(
// CHECK:         %[[RES:.*]] = arith.divsi %{{.*}}, %{{.*}} : i32
llvm.func @sdiv(%lhs: i32, %rhs: i32) -> i32 {
  %0 = llvm.sdiv %lhs, %rhs : i32
  llvm.return %0 : i32
}

// CHECK-LABEL: llvm.func @sdiv_exact(
// CHECK:         %[[RES:.*]] = arith.divsi %{{.*}}, %{{.*}} exact : i32
llvm.func @sdiv_exact(%lhs: i32, %rhs: i32) -> i32 {
  %0 = llvm.sdiv exact %lhs, %rhs : i32
  llvm.return %0 : i32
}

// CHECK-LABEL: llvm.func @udiv(
// CHECK:         %[[RES:.*]] = arith.divui %{{.*}}, %{{.*}} : i32
llvm.func @udiv(%lhs: i32, %rhs: i32) -> i32 {
  %0 = llvm.udiv %lhs, %rhs : i32
  llvm.return %0 : i32
}

// CHECK-LABEL: llvm.func @icmp(
// CHECK:         %[[RES:.*]] = arith.cmpi slt, %{{.*}}, %{{.*}} : i32
llvm.func @icmp(%lhs: i32, %rhs: i32) -> i1 {
  %0 = llvm.icmp "slt" %lhs, %rhs : i32
  llvm.return %0 : i1
}

// CHECK-LABEL: llvm.func @icmp_all_predicates(
// CHECK:         arith.cmpi eq
// CHECK:         arith.cmpi ne
// CHECK:         arith.cmpi slt
// CHECK:         arith.cmpi sle
// CHECK:         arith.cmpi sgt
// CHECK:         arith.cmpi sge
// CHECK:         arith.cmpi ult
// CHECK:         arith.cmpi ule
// CHECK:         arith.cmpi ugt
// CHECK:         arith.cmpi uge
llvm.func @icmp_all_predicates(%lhs: i32, %rhs: i32) -> i1 {
  %0 = llvm.icmp "eq" %lhs, %rhs : i32
  %1 = llvm.icmp "ne" %lhs, %rhs : i32
  %2 = llvm.icmp "slt" %lhs, %rhs : i32
  %3 = llvm.icmp "sle" %lhs, %rhs : i32
  %4 = llvm.icmp "sgt" %lhs, %rhs : i32
  %5 = llvm.icmp "sge" %lhs, %rhs : i32
  %6 = llvm.icmp "ult" %lhs, %rhs : i32
  %7 = llvm.icmp "ule" %lhs, %rhs : i32
  %8 = llvm.icmp "ugt" %lhs, %rhs : i32
  %9 = llvm.icmp "uge" %lhs, %rhs : i32
  %r = llvm.and %0, %9 : i1
  llvm.return %r : i1
}

// Vector operands are supported as well.

// CHECK-LABEL: llvm.func @vector_ops(
// CHECK:         %[[ADD:.*]] = arith.addi %{{.*}}, %{{.*}} : vector<4xi32>
// CHECK:         %[[CMP:.*]] = arith.cmpi ule, %[[ADD]], %{{.*}} : vector<4xi32>
llvm.func @vector_ops(%lhs: vector<4xi32>, %rhs: vector<4xi32>) -> vector<4xi1> {
  %0 = llvm.add %lhs, %rhs : vector<4xi32>
  %1 = llvm.icmp "ule" %0, %rhs : vector<4xi32>
  llvm.return %1 : vector<4xi1>
}

// Pointer comparisons are not representable in the arith dialect and must be
// preserved.

// CHECK-LABEL: llvm.func @icmp_ptr(
// CHECK:         %[[RES:.*]] = llvm.icmp "ult" %{{.*}}, %{{.*}} : !llvm.ptr
llvm.func @icmp_ptr(%lhs: !llvm.ptr, %rhs: !llvm.ptr) -> i1 {
  %0 = llvm.icmp "ult" %lhs, %rhs : !llvm.ptr
  llvm.return %0 : i1
}

// CHECK-LABEL: llvm.func @icmp_ptr_vector(
// CHECK:         %[[RES:.*]] = llvm.icmp "ult" %{{.*}}, %{{.*}} : vector<4x!llvm.ptr>
llvm.func @icmp_ptr_vector(%lhs: vector<4x!llvm.ptr>, %rhs: vector<4x!llvm.ptr>) -> vector<4xi1> {
  %0 = llvm.icmp "ult" %lhs, %rhs : vector<4x!llvm.ptr>
  llvm.return %0 : vector<4xi1>
}

// Operations other than the supported ones are preserved.

// CHECK-LABEL: llvm.func @untouched(
// CHECK:         llvm.urem
// CHECK:         llvm.and
llvm.func @untouched(%lhs: i32, %rhs: i32) -> i32 {
  %0 = llvm.urem %lhs, %rhs : i32
  %1 = llvm.and %0, %rhs : i32
  llvm.return %1 : i32
}
