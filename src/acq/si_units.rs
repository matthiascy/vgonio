macro impl_serialization($t:ty where $unit:ident: $unit_trait:ident, #[$doc:meta]) {
    #[$doc]
    impl<$unit: $unit_trait> serde::Serialize for $t {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            serializer.serialize_str(&format!("{}", self))
        }
    }
}

macro impl_ops {
    ($($op:ident),* for $t:ident where $unit_a:ident, $unit_b:ident: $unit_trait:ident) => {
        paste::paste! {
            $(
                impl<$unit_a: $unit_trait, $unit_b: $unit_trait> core::ops::$op<$t<$unit_b>> for $t<$unit_a>
                where $t<$unit_a>: From<$t<$unit_b>>
                {
                    type Output = $t<$unit_a>;

                    fn [<$op:lower>](self, rhs: $t<$unit_b>) -> Self::Output {
                        let rhs: $t<$unit_a> = rhs.into();
                        $t {
                            value: self.value.[<$op:lower>](rhs.value),
                            unit: core::marker::PhantomData,
                        }
                    }
                }
            )*
        }
    },
    // ($($op:ident),* for $t:ident where A, B: $unit_trait:ident => $output:ty) => {
    //     paste::paste! {
    //         $(
    //             impl<A: $unit_trait, B: $unit_trait> core::ops::$op<$t<B>> for $t<A>
    //             where $t<A>: From<$t<B>>
    //             {
    //                 type Output = $output;
    //
    //                 fn [<$op:lower>](self, rhs: $t<B>) -> Self::Output {
    //                     let rhs: $t<A> = rhs.into();
    //                     $t {
    //                         value: self.value.[<$op:lower>](rhs.value),
    //                         unit: core::marker::PhantomData,
    //                     }
    //                 }
    //             }
    //         )*
    //     }
    // }
}

macro impl_ops_with_f32($($op:ident),* for $t:ident where A: $unit_trait:ident) {
    paste::paste! {
        $(
            impl<A: $unit_trait> core::ops::$op<f32> for $t<A> {
                type Output = $t<A>;

                fn [<$op:lower>](self, rhs: f32) -> Self::Output {
                    $t {
                        value: self.value.[<$op:lower>](rhs),
                        unit: core::marker::PhantomData,
                    }
                }
            }

            impl<A: $unit_trait> core::ops::[<$op Assign>]<f32> for $t<A> {
                fn [<$op:lower _assign>](&mut self, rhs: f32) {
                    self.value.[<$op:lower _assign>](rhs);
                }
            }
        )*
    }
}

macro impl_ops_assign($($op:ident),* for $t:ident where A, B: $unit_trait:ident) {
    paste::paste! {
        $(
            impl<A: $unit_trait, B: $unit_trait> core::ops::$op<$t<B>> for $t<A>
            where $t<A>: From<$t<B>>
            {
                fn [<$op:snake>](&mut self, rhs: $t<B>) {
                    let rhs: $t<A> = rhs.into();
                    self.value.[<$op:snake>](rhs.value);
                }
            }
        )*
    }
}

macro forward_f32_methods($($name:ident),+) {
    $(
        #[inline(always)]
        pub fn $name(self) -> Self {
            Self {
                value: self.value.$name(),
                unit: core::marker::PhantomData,
            }
        }
    )+
}

mod angle;
mod length;
mod solid_angle;

pub use angle::*;
pub use length::*;
pub use solid_angle::*;

fn findr_first_non_ascii_alphabetic(s: &[u8]) -> Option<usize> {
    let mut i = s.len();
    while i > 0 {
        if s[i - 1].is_ascii_alphabetic() {
            i -= 1;
        } else {
            return Some(i);
        }
    }
    None
}
