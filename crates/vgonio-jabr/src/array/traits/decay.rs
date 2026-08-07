/// Trait to obtain the decayed type of any type.
///
/// Mostly used to remove references and constness from types.
pub const trait Decay {
    type Output;
}

const impl<T> Decay for T {
    default type Output = T;
}

const impl<T> Decay for &T {
    type Output = <T as Decay>::Output;
}

const impl<T> Decay for &mut T {
    type Output = <T as Decay>::Output;
}

const impl<T> Decay for *const T {
    type Output = T;
}

const impl<T> Decay for *mut T {
    type Output = T;
}
