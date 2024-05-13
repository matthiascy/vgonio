/// Trait to obtain the decayed type of any type.
///
/// Mostly used to remove references and constness from types.
#[const_trait]
pub trait Decay {
    type Output;
}

impl<T> const Decay for T {
    default type Output = T;
}

impl<T> const Decay for &T {
    type Output = <T as Decay>::Output;
}

impl<T> const Decay for &mut T {
    type Output = <T as Decay>::Output;
}

impl<T> const Decay for *const T {
    type Output = T;
}

impl<T> const Decay for *mut T {
    type Output = T;
}
