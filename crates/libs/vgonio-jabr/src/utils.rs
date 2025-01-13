use std::marker::PhantomData;

pub struct Assert<const COND: bool> {}

pub trait IsTrue {}

impl IsTrue for Assert<true> {}

pub trait IsFalse {}

pub struct AssertSameType<A, B>(PhantomData<(A, B)>);

impl<T> IsTrue for AssertSameType<T, T> {}
