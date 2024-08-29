use crate::Vec3;

// TODO: constify
/// Reflects a vector 'wi' about the given normal 'n'.
///
/// # Arguments
///
/// * `wi` - Incident direction (not necessarily normalized), pointing towards
///   the boundary between two media (ends up on the point of incidence).
///
/// * `n` - The normal vector (must be normalized).
///
/// # Returns
///
/// The reflected vector.
///
/// # Notes
///
/// The normal vector is assumed to be always pointing towards the medium where
/// the incident ray is coming from.
#[must_use]
#[inline(always)]
pub fn reflect(wi: &Vec3, n: &Vec3) -> Vec3 {
    debug_assert!(
        wi.dot(n) <= 0.0,
        "The incident direction must be pointing towards the boundary."
    );
    debug_assert!(
        (n.norm_sqr() - 1.0) < 1e-8,
        "The normal vector must be normalized."
    );
    wi - 2.0 * wi.dot(n) * n
}

/// Reflects a vector 'wi' about the given normal 'n' with the known cosine of
/// incident angle.
///
/// # Arguments
///
/// * `wi` - Incident direction (not necessarily normalized), pointing towards
///   the boundary between two media (ends up on the point of incidence).
/// * `n` - The normal vector (must be normalized).
/// * `cos` - The cosine of the incident angle, should always be positive; this
///   is *NOT* the angle between the two vectors. It should be the absolute
///   value of the cosine of the angle between 'wi' and 'n'.
///
/// # Returns
///
/// The reflected vector.
///
/// # Notes
///
/// The direction of the incident vector `wi` is determined from ray's origin to
/// the point of incidence rather than the other way around.
#[must_use]
#[inline(always)]
pub fn reflect_cos(wi: &Vec3, n: &Vec3, cos: f64) -> Vec3 {
    debug_assert!(
        wi.dot(n) <= 0.0,
        "The incident direction must be pointing towards the boundary."
    );
    debug_assert!(
        cos >= 0.0,
        "The cosine of the incident angle must be positive."
    );
    debug_assert!(
        (n.norm_sqr() - 1.0).abs() < 1e-8,
        "The normal vector must be normalized."
    );
    wi + 2.0 * cos * n
}

/// Result of refraction.
pub enum Refracted {
    /// The total internal reflection occurred.
    Reflection(Vec3),
    /// The refraction occurred.
    Refraction {
        /// Direction of the refracted ray. It's the direction of the
        /// transmitted ray.
        dir_t: Vec3,
        /// The cosine of the transmitted angle. It's the absolute value of the
        /// cosine of the angle between the transmitted ray and the normal
        /// vector (without knowing normal's side).
        cos_t: f64,
    },
}

/// Refracts the incident vector 'wi' about the given normal 'n' and the
/// refractive indices of the two media.
///
/// # Notes
///
/// Use this function when you are sure that the incident vector `wi` is at the
/// same side of the surface as the normal vector `n`, i.e., the incident vector
/// `wi` is coming from the medium where the normal vector `n` is pointing.
/// Hence, `eta_i` should be the refractive index of the medium where the
/// incident ray is coming from (incident medium) and `eta_o` should be the
/// refractive index of the medium where the refracted ray is going to (inside
/// transmitted medium). If you are not sure about the side of incident vector
/// `wi`, use [`refract2`] instead.
///
/// # Arguments
///
/// * `wi` - Incident direction (must be normalized), pointing towards the
///   boundary between two media (ends up on the point of incidence).
/// * `n` - The normal vector (must be normalized), always pointing towards the
///   medium where the incident ray is coming from.
/// * `eta` -  The relative index of refraction, which is the ratio of the
///   refractive index of outside medium (where `n` is pointing, primary medium
///   or incident medium) over the refractive index of the inside medium
///   (secondary medium, transmitted medium).
///
/// # Returns
///
/// The result of refraction [`Refracted`].
#[must_use]
#[inline(always)]
pub fn refract(wi: &Vec3, n: &Vec3, eta: f64) -> Refracted {
    debug_assert!(
        wi.dot(n) <= 0.0,
        "The incident direction must be pointing towards the boundary."
    );
    debug_assert!(
        (n.norm_sqr() - 1.0).abs() < 1e-8,
        "The normal vector must be normalized."
    );
    debug_assert!(
        (wi.norm_sqr() - 1.0).abs() < 1e-8,
        "The incident direction must be normalized."
    );
    debug_assert!(eta > 0.0, "The refractive index must be positive.");
    let cos_i = -wi.dot(n);
    refract_cos(wi, n, cos_i, eta)
}

/// Refracts the incident vector 'wi' about the given normal 'n' with the known
/// cosine of incident angle.
///
/// # Notes
///
/// Use this function when you are sure that the incident vector `wi`
/// is at the same side of the boundary as the normal vector `n`; the relative
/// index of refraction should be the ratio of the refractive index of the
/// medium where the incident ray is coming from (incident medium) to the
/// refractive index of the medium where the refracted ray is going to (inside
/// transmitted medium).
///
/// # Arguments
///
/// * `wi` - Incident direction (must be normalized), pointing towards the
///   boundary between two media (ends up on the point of incidence).
/// * `n` - The normal vector (must be normalized), always pointing towards the
///   medium where the incident ray is coming from.
/// * `cos` - The cosine of the incident angle, should always be positive; this
///   is *NOT* the angle between the two vectors. It should be the absolute
///   value of the cosine of the angle between 'wi' and 'n'.
/// * `eta` - The relative index of refraction, which is the ratio of the
///   refractive index of outside medium (where `n` is pointing, primary medium
///   or incident medium) over the refractive index of the inside medium
///   (secondary medium, transmitted medium), i.e., `eta = eta_i / eta_t`, where
///   `eta_i` is the refractive index of the incident medium and `eta_t` is the
///   refractive index of the transmitted medium.
/// * `cos` - The cosine of the incident angle, should always be positive; this
///   is *NOT* the angle between the two vectors. It should be the absolute
///   value of the cosine of the angle between 'wi' and 'n'.
///
/// # Returns
///
/// The result of refraction [`Refracted`].
#[must_use]
#[inline(always)]
pub fn refract_cos(wi: &Vec3, n: &Vec3, cos: f64, eta: f64) -> Refracted {
    debug_assert!(
        wi.dot(n) <= 0.0,
        "The incident direction must be pointing towards the boundary."
    );
    debug_assert!(
        cos >= 0.0,
        "The cosine of the incident angle must be positive."
    );
    debug_assert!(
        (n.norm_sqr() - 1.0).abs() < 1e-8,
        "The normal vector must be normalized."
    );
    debug_assert!(
        (wi.norm_sqr() - 1.0).abs() < 1e-8,
        "The incident direction must be normalized."
    );
    debug_assert!(
        eta > 0.0,
        "The relative index of refraction must be positive."
    );

    let cos_t_sqr = 1.0 - eta * eta * (1.0 - cos * cos);

    if cos_t_sqr < 0.0 {
        Refracted::Reflection(reflect_cos(wi, n, cos))
    } else {
        Refracted::Refraction {
            dir_t: eta * wi + (eta * cos - cos_t_sqr.sqrt()) * n,
            cos_t: cos_t_sqr.sqrt(),
        }
    }
}

/// Refracts the incident vector 'wi' about the given normal 'n' and the
/// refractive indices of the two media.
///
/// # Notes
///
/// Use this function when you are not sure about the side of incident vector
/// `wi` and the normal vector `n`. If you are sure about the side of incident
/// vector `wi`, use [`refract`] or [`refract_cos`] instead.
///
/// The normal vector `n` is not necessarily pointing towards the medium where
/// the incident ray is coming from. Thus, we need to check the side of the
/// incident vector `wi` and the normal vector `n` by calculating the dot
/// product between them and then decide whether the incident vector `wi` is at
/// the same side of the boundary as the normal vector `n` or at the opposite
/// side of the boundary as the normal vector `n`.
///
/// # Arguments
///
/// * `wi` - Incident direction (must be normalized), pointing towards the
///   boundary between two media (ends up on the point of incidence).
/// * `n` - The normal vector (must be normalized).
/// * `eta_o` - The refractive index of the outside medium (where `n` is
///  pointing, primary medium or incident medium).
/// * `eta_i` - The refractive index of the inside medium (transmitted medium).
#[must_use]
#[inline(always)]
pub fn refract2(wi: &Vec3, n: &Vec3, eta_o: f64, eta_i: f64) -> Refracted {
    debug_assert!(
        (n.norm_sqr() - 1.0).abs() < 1e-8,
        "The normal vector must be normalized."
    );
    debug_assert!(
        (wi.norm_sqr() - 1.0).abs() < 1e-8,
        "The incident direction must be normalized."
    );
    debug_assert!(
        eta_i > 0.0,
        "The refractive index of the transmitted medium must be positive."
    );
    debug_assert!(
        eta_o > 0.0,
        "The refractive index of the incident medium must be positive."
    );
    let cos_i = wi.dot(n);
    if cos_i < 0.0 {
        // The incident vector is at the same side of the boundary as the normal
        // vector.
        refract_cos(wi, n, -cos_i, eta_o / eta_i)
    } else {
        // The incident vector is at the opposite side of the boundary as the
        // normal vector.
        refract_cos(wi, &(-n), cos_i, eta_i / eta_o)
    }
}

/// Computes the Schlick's approximation for the Fresnel reflectance(specular
/// reflection factor).
///
/// # Arguments
///
/// * `cos_i` - The absolute value of the cosine of the incident angle.
/// * `eta_i` - The refractive index of the incident medium.
/// * `eta_t` - The refractive index of the transmitted medium.
pub fn reflectance_schlick(cos_i: f64, eta_i: f64, eta_t: f64) -> f64 {
    debug_assert!(
        cos_i >= 0.0 && cos_i <= 1.0,
        "The cosine of the incident angle must be in the range [0, 1]."
    );
    debug_assert!(
        eta_i > 0.0,
        "The refractive index of the incident medium must be positive."
    );
    debug_assert!(
        eta_t > 0.0,
        "The refractive index of the transmitted medium must be positive."
    );
    let r0 = ((eta_i - eta_t) / (eta_i + eta_t)).powi(2);
    r0 + (1.0 - r0) * (1.0 - cos_i).powi(5)
}
