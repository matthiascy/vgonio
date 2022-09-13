macro_rules! impl_serialization {
    ( $t:ty where $unit:ident: $unit_trait:ident, #[$doc:meta]) => {
        #[$doc]
        impl<$unit: $unit_trait> serde::Serialize for $t {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                serializer.serialize_str(&format!("{}", self))
            }
        }
    };
}

macro_rules! impl_ops {
    ($($op:ident),* for $t:ident where A, B: $unit_trait:ident) => {
        paste! {
            $(
                impl<A: $unit_trait, B: $unit_trait> $op<$t<B>> for $t<A>
                    where $t<A>: From<$t<B>>
                {
                    type Output = $t<A>;

                    fn [<$op:lower>](self, rhs: $t<B>) -> Self::Output {
                        let rhs: $t<A> = rhs.into();
                        $t {
                            value: self.value.[<$op:lower>](rhs.value),
                            unit: PhantomData,
                        }
                    }
                }
            )*
        }
    };
}

macro_rules! impl_ops_with_f32 {
    ($($op:ident),* for $t:ident where A: $unit_trait:ident) => {
        paste! {
            $(
                impl<A: $unit_trait> $op<f32> for $t<A> {
                    type Output = $t<A>;

                    fn [<$op:lower>](self, rhs: f32) -> Self::Output {
                        $t {
                            value: self.value.[<$op:lower>](rhs),
                            unit: PhantomData,
                        }
                    }
                }

                impl<A: $unit_trait> [<$op Assign>]<f32> for $t<A> {
                    fn [<$op:lower _assign>](&mut self, rhs: f32) {
                        self.value.[<$op:lower _assign>](rhs);
                    }
                }
            )*
        }
    };
}

macro_rules! impl_ops_assign {
    ($($op:ident),* for $t:ident where A, B: $unit_trait:ident) => {
        paste! {
            $(
            impl<A: $unit_trait, B: $unit_trait> $op<$t<B>> for $t<A>
            where $t<A>: From<$t<B>>
            {
                fn [<$op:snake>](&mut self, rhs: $t<B>) {
                    let rhs: $t<A> = rhs.into();
                    self.value.[<$op:snake>](rhs.value);
                }
            }
            )*
        }
    };
}

mod angle;
mod length;

pub use angle::*;
pub use length::*;

fn findr_first_ascii_alphabetic(s: &[u8]) -> Option<usize> {
    let mut i = s.len() - 1;
    while i > 0 {
        if s[i].is_ascii_alphabetic() {
            i -= 1;
        } else {
            return Some(i);
        }
    }
    None
}
