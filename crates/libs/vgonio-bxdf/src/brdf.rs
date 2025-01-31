//! Brdf models, especially the microfacet-based ones.
use vgonio_core::math::{Mat3, Sph2, Vec3};

pub mod analytical;
pub mod lambert;
pub mod measured;

#[rustfmt::skip]
/// Converts the incident and outgoing direction to the half and difference
/// vector.
///
/// $$
/// \begin{cases}
///   \mathbf{\h} = \frac{\mathbf{\i} + \mathbf{\o}}{||\mathbf{\i} + \mathbf{\o}||} \\\\
///   \mathbf{\d} = Rot_y(-\theta_h) \cdot Rot_z(-\phi_h) \cdot \mathbf{\i}
/// \end{cases}
/// $$
///
/// where $\theta_h$ and $\phi_h$ are the polar and azimuthal angles of the half-vector.
///
/// # Arguments
///
/// * `vi` - The incident direction.
/// * `vo` - The outgoing direction.
///
/// # Returns
///
/// (half, difference)
pub fn io2hd(vi: &Vec3, vo: &Vec3) -> (Vec3, Vec3) {
    let h = (*vi + *vo).normalize();
    let wh = Sph2::from_cartesian(h);
    let rot_y = Mat3::from_rotation_y(-wh.theta.as_f32());
    let rot_z = Mat3::from_rotation_z(-wh.phi.as_f32());
    let d = rot_y * rot_z * *vi;
    (h, d)
}

/// Converts the incident and outgoing direction to half-vector and difference
/// vector in spherical coordinates.
///
/// See [`io2hd`] for the conversion between the incident and outgoing direction
/// and the half and difference vector in Cartesian coordinates.
///
/// # Arguments
///
/// * `wi` - The incident direction in spherical coordinates.
/// * `wo` - The outgoing direction in spherical coordinates.
///
/// # Returns
///
/// (half, difference)
pub fn io2hd_sph(wi: &Sph2, wo: &Sph2) -> (Sph2, Sph2) {
    let (h, d) = io2hd(&wi.to_cartesian(), &wo.to_cartesian());
    (Sph2::from_cartesian(h), Sph2::from_cartesian(d))
}

#[rustfmt::skip]
/// Converts the half and difference vectors to the incident and outgoing
/// direction.
///
/// $$
/// \begin{cases}
///   \mathbf{\i} = Rot_z(+\phi_h) \cdot Rot_y(+\theta_h) \cdot \mathbf{\d} \\\\
///   \mathbf{\o} = 2 \cdot (\mathbf{\i} \cdot \mathbf{h})\mathbf{\h} - \mathbf{\i}
/// \end{cases}
/// $$
///
/// where $\theta_h$ and $\phi_h$ are the polar and azimuthal angles of the half-vector.
///
/// # Arguments
///
/// * `vh` - The half vector.
/// * `vd` - The difference vector.
///
/// # Returns
///
/// (incident, outgoing)
pub fn hd2io(vh: &Vec3, vd: &Vec3) -> (Vec3, Vec3) {
    let wh = Sph2::from_cartesian(*vh);
    let phi_h = wh.phi;
    let theta_h = wh.theta;
    let rot_y = Mat3::from_rotation_y(theta_h.as_f32());
    let rot_z = Mat3::from_rotation_z(phi_h.as_f32());
    let vi = rot_z * rot_y * *vd;
    let vo = 2.0 * vi.dot(*vh) * *vh - vi;
    (vi, vo)
}

/// Converts the half and difference vectors to the incident and outgoing
/// direction in spherical coordinates.
///
/// # Arguments
///
/// * `wh` - The half vector in spherical coordinates.
/// * `wd` - The difference vector in spherical coordinates.
///
/// # Returns
///
/// (incident, outgoing)
pub fn hd2io_sph(wh: &Sph2, wd: &Sph2) -> (Sph2, Sph2) {
    let (i, o) = hd2io(&wh.to_cartesian(), &wd.to_cartesian());
    (Sph2::from_cartesian(i), Sph2::from_cartesian(o))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vgonio_core::units::Radians;

    #[track_caller]
    fn assert_vec3_eq(v1: &Vec3, v2: &Vec3) {
        assert!((v1.x - v2.x).abs() < 1e-6, "x: {} != {}", v1.x, v2.x);
        assert!((v1.y - v2.y).abs() < 1e-6, "y: {} != {}", v1.y, v2.y);
        assert!((v1.z - v2.z).abs() < 1e-6, "z: {} != {}", v1.z, v2.z);
    }

    #[test]
    fn test_io2hd() {
        let i = Vec3::new(1.0, 0.0, 0.0);
        let o = Vec3::new(0.0, 1.0, 0.0);
        let (h, d) = io2hd(&i, &o);
        assert_eq!(h, Vec3::new(1.0, 1.0, 0.0).normalize());
        let expected_d = Vec3::new(0.0, -1.0, 1.0).normalize();
        assert_vec3_eq(&d, &expected_d);

        {
            let i_converted = hd2io(&h, &d).0;
            assert_vec3_eq(&i_converted, &i);
            let o_converted = hd2io(&h, &d).1;
            assert_vec3_eq(&o_converted, &o);
        }

        let wi = Sph2::new(Radians::PI / 4.0, Radians::PI / 3.0);
        let wo = Sph2::new(Radians::PI / 3.0, Radians::PI / 2.0);
        let (h, d) = io2hd_sph(&wi, &wo);
        let h_expected = Vec3::new(0.182143, 0.761639, 0.621876);
        let d_expected = Vec3::new(
            -0.13223252786478978393,
            -0.20142696029304382743,
            0.97053682992537293472,
        );
        assert_vec3_eq(&h.to_cartesian(), &h_expected);
        assert_vec3_eq(&d.to_cartesian(), &d_expected);

        {
            let i_converted = hd2io_sph(&h, &d).0;
            assert_vec3_eq(&i_converted.to_cartesian(), &wi.to_cartesian());
            let o_converted = hd2io_sph(&h, &d).1;
            assert_vec3_eq(&o_converted.to_cartesian(), &wo.to_cartesian());
        }
    }
}
