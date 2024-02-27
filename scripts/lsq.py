import struct
import sys

import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy as np
from scipy.special import erf
import matplotlib.tri as mtri


def read_brdf_data(filename):
    """
    Read the BRDF data from a file and return it as a numpy array.
    """
    with (open(filename, 'rb') as f):
        # Read the metadata
        metadata = f.read(141)
        file_format = metadata[0:4]
        if file_format != b'VGMO':
            raise Exception(
                f'Invalid file format, the file does not contain the correct data: BRDF required. {file_format}')
        (major, minor, patch, _) = struct.unpack('<BBBB', metadata[4:8])
        (file_size,) = struct.unpack('<I', metadata[8:12])
        print(f"file size: {file_size}")
        timestamp = metadata[12:44]
        print(f"timestamp: {timestamp}")
        (sample_size,) = struct.unpack('<B', metadata[44:45])
        print(f"sample size: {sample_size}")
        is_binary = metadata[45:46] == b'!'
        print(f"is binary: {is_binary} ({metadata[45:46]})")
        (_compression_type,) = struct.unpack('<B', metadata[46:47])
        is_compressed = _compression_type != 0
        compression_type = 'none' if not is_compressed else 'zlib' if _compression_type == 1 else 'gzip' if _compression_type == 2 else 'unknown'
        print(f"is compressed: {is_compressed} ({compression_type})")
        (mtype,) = struct.unpack('<B', metadata[48:49])
        measurement_type = 'BRDF' if mtype == 0 else 'ADF' if mtype == 1 else 'MSF' if mtype == 2 else 'unknown'
        print(f"measurement type: {measurement_type} ({mtype})")
        _bsdf_type = metadata[49:50]
        bsdf_type = 'brdf' if _bsdf_type == b'0' \
            else 'btdf' if _bsdf_type == b'1' \
            else 'bssdf' if _bsdf_type == b'2' \
            else 'bssrdf' if _bsdf_type == b'3' \
            else 'bsstdf' if _bsdf_type == b'4' \
            else 'unknown'
        _medium_i = metadata[50:53]
        incident_medium = 'vacuum' if _medium_i == b'vac' else 'air' if _medium_i == b'air' else 'aluminum' if _medium_i == b'al\x00' else 'copper' if _medium_i == b'cu\x00' else 'known'
        print(f"incident medium: {incident_medium} ({_medium_i})")
        _medium_t = metadata[53:56]
        transmitted_medium = 'vacuum' if _medium_t == b'vac' else 'air' if _medium_t == b'air' else 'aluminum' if _medium_t == b'al\x00' else 'copper' if _medium_t == b'cu\x00' else 'known'
        print(f"transmitted medium: {transmitted_medium} ({_medium_t})")
        _sim_method = metadata[56:57]
        simulation_method = 'grid-rt' if _sim_method == b'\x00' else 'embree-rt' if _sim_method == b'\x01' else 'optix-rt' if _sim_method == b'\x02' else 'wave' if _sim_method == b'\x03' else 'unknown'
        print(f"simulation method: {simulation_method} ({_sim_method})")
        (n_rays,) = struct.unpack('<I', metadata[57:61])
        print(f"number of rays: {n_rays}")
        (max_bounces,) = struct.unpack('<I', metadata[61:65])
        print(f"max bounces: {max_bounces}")
        # longitudes
        (emitter_start_phi, emitter_end_phi, emitter_step_phi) = struct.unpack('<fff', metadata[65:77])
        print(
            f"emitter start phi: {np.degrees(emitter_start_phi)}, end phi: {np.degrees(emitter_end_phi)}, step phi: {np.degrees(emitter_step_phi)}")
        (emitter_phi_step_count,) = struct.unpack('<I', metadata[77:81])
        print(f"emitter phi step count: {emitter_phi_step_count}")
        # colatitudes
        (emitter_start_theta, emitter_end_theta, emitter_step_theta) = struct.unpack('<fff', metadata[81:93])
        print(
            f"emitter start theta: {np.degrees(emitter_start_theta)}, end theta: {np.degrees(emitter_end_theta)}, step theta: {np.degrees(emitter_step_theta)}")
        (emitter_theta_step_count,) = struct.unpack('<I', metadata[93:97])
        print(f"emitter theta step count: {emitter_theta_step_count}")
        wi_count = emitter_phi_step_count * emitter_theta_step_count
        print(f"wi count: {wi_count}")
        # wavelengths
        (start_wavelength, end_wavelength, step_wavelength) = struct.unpack('<fff', metadata[97:109])
        print(
            f"start wavelength: {start_wavelength}, end wavelength: {end_wavelength}, step wavelength: {step_wavelength}")
        (wavelength_count,) = struct.unpack('<I', metadata[109:113])  # TODO: check if this is correct
        print(f"wavelength count: {wavelength_count}")
        (_domain,) = struct.unpack('<I', metadata[113:117])
        domain = 'whole' if _domain == 0 else 'upper' if _domain == 1 else 'lower' if _domain == 2 else 'unknown'
        print(f"domain: {domain} ({_domain})")
        (partition_,) = struct.unpack('<I', metadata[117:121])
        partition = 'beckers' if partition_ == 0 else 'tregenza' if partition_ == 1 else 'equal-angle' if partition_ == 2 else 'unknown'
        print(f"partition: {partition} ({partition_})")
        (parition_precision_theta, parition_precision_phi) = struct.unpack('<ff', metadata[121:129])
        print(
            f"partition precision theta: {np.degrees(parition_precision_theta)}, phi: {np.degrees(parition_precision_phi)}")
        (n_rings, n_patches) = struct.unpack('<II', metadata[129:137])  # TODO: check if this is correct
        print(f"n_rings: {n_rings}, n_patches: {n_patches}")
        (data_sample_type_,) = struct.unpack('<I', metadata[137:141])
        if data_sample_type_ == 1:
            data_sample_type = 'bsdf only'
        elif data_sample_type_ == 0:
            data_sample_type = 'full data'
        print(f"data sample type: {data_sample_type} ({data_sample_type_})")
        rings_info = f.read(n_rings * 20)
        rings = []
        for i in range(n_rings):
            (theta_min, theta_max, phi_step_size, r_n_patches, index_offset) = \
                struct.unpack('<ffIII', rings_info[i * 20:i * 20 + 20])
            rings.append((theta_min, theta_max, phi_step_size, r_n_patches, index_offset))

        # Calculate the outgoing direction for each patch in each ring
        wos = np.zeros((n_patches, 2))
        print(wos.shape)
        for ring in rings:
            (theta_min, theta_max, phi_step_size, r_n_patches, index_offset) = ring
            theta = (theta_min + theta_max) * 0.5
            for i in range(r_n_patches):
                phi_min = i * phi_step_size
                phi_max = (i + 1) * phi_step_size
                phi = (phi_min + phi_max) * 0.5
                wos[index_offset + i, :] = [theta, phi]

        snapshot_size = n_patches * sample_size * wavelength_count * 4 + 8

        print(
            f"file size: {file_size} =? {141 + n_rings * 20 + wi_count * (n_patches * sample_size * wavelength_count * 4 + 8)}")

        # Read the data
        wis = np.zeros((wi_count, 2))
        snapshots = np.zeros((wi_count, n_patches))
        print('snapshot shape: ', snapshots.shape)
        if is_binary:
            if compression_type == 'none':
                for i in range(wi_count):
                    # Read the data as binary
                    (wi_theta, wi_phi) = struct.unpack('<ff', f.read(8))
                    wis[i] = (wi_theta, wi_phi)
                    full_snapshot = np.frombuffer(
                        f.read(n_patches * wavelength_count * sample_size),
                        dtype=np.float32,
                        count=n_patches * wavelength_count,
                    ).reshape(1, n_patches, wavelength_count)
                    snapshots[i][:] = full_snapshot[:, :, 0][:]
            else:
                raise Exception('Not implemented yet.')
        return wis, wos, snapshots


def trowbridge_reitz_ndf_iso(alpha, cos_theta_h):
    """
    Calculate the Trowbridge-Reitz BRDF model with the given roughness
    and the given incident and outgoing directions.
    """
    cos_theta_h2 = cos_theta_h * cos_theta_h
    cos_theta_h4 = cos_theta_h2 * cos_theta_h2
    if cos_theta_h4 < 1e-8:
        return 0.0
    tan_theta_h2 = (1 - cos_theta_h2) / cos_theta_h2
    if tan_theta_h2 == np.inf:
        return 0.0
    alpha2 = alpha * alpha
    return alpha2 / (np.pi * cos_theta_h2 * cos_theta_h2 * (alpha2 + tan_theta_h2) ** 2)


def beckmann_ndf_iso(alpha, cos_theta_h):
    alpha2 = alpha * alpha
    cos_theta_h2 = cos_theta_h * cos_theta_h
    cos_theta_h4 = cos_theta_h2 * cos_theta_h2
    if cos_theta_h4 < 1e-8:
        return 0.0
    tan_theta_h2 = (1 - cos_theta_h2) / cos_theta_h2
    return np.exp(-tan_theta_h2 / alpha2) / (np.pi * alpha2 * cos_theta_h4)


def fresnel(cos_i_abs, eta_i, eta_t, k_t):
    """
    Calculate the Fresnel reflection coefficient.
    """
    eta = eta_t / eta_i
    k = k_t / eta_i
    cos_i2 = cos_i_abs * cos_i_abs
    sin_i2 = 1 - cos_i2
    eta2 = eta * eta
    k2 = k * k
    t0 = eta2 - k2 - sin_i2
    a2_plus_b2 = np.sqrt(t0 * t0 + 4.0 * k_t * k_t * eta_t * eta_t)
    t1 = a2_plus_b2 + cos_i2
    a = np.sqrt(0.5 * (a2_plus_b2 + t0))
    t2 = 2.0 * a * cos_i_abs
    rs = (t1 - t2) / (t1 + t2)
    t3 = a2_plus_b2 * cos_i2 + sin_i2 * sin_i2
    t4 = t2 * sin_i2
    rp = rs * (t3 - t4) / (t3 + t4)

    return 0.5 * (rp + rs)


def trowbridge_reitz_geom_iso(alpha, cos_theta_hi, cos_theta_ho):
    """
    Calculate the Trowbridge-Reitz geometric attenuation factor.
    """
    alpha2 = alpha * alpha
    cos_theta_hi2 = cos_theta_hi * cos_theta_hi
    cos_theta_ho2 = cos_theta_ho * cos_theta_ho
    tan_theta_hi2 = (1 - cos_theta_hi2) / cos_theta_hi2
    tan_theta_ho2 = (1 - cos_theta_ho2) / cos_theta_ho2
    return 2 / (1 + np.sqrt(1 + alpha2 * tan_theta_hi2)) * 2 / (1 + np.sqrt(1 + alpha2 * tan_theta_ho2))


def beckmann_geom_iso(alpha, cos_theta_hi, cos_theta_ho):
    tan_theta_hi = np.sqrt(1 - cos_theta_hi * cos_theta_hi) / cos_theta_hi
    tan_theta_ho = np.sqrt(1 - cos_theta_ho * cos_theta_ho) / cos_theta_ho
    ai = 1.0 / (alpha * tan_theta_hi)
    ao = 1.0 / (alpha * tan_theta_ho)
    erf_ai = erf(ai)
    erf_ao = erf(ao)
    ei = np.exp(-ai * ai)
    eo = np.exp(-ao * ao)
    gi = 2.0 / (1.0 + erf_ai + ei / (np.sqrt(np.pi) * ai))
    go = 2.0 / (1.0 + erf_ao + eo / (np.sqrt(np.pi) * ao))
    return gi * go


def trowbridge_reitz_iso(alpha, wi: np.ndarray, wo: np.ndarray):
    """
    Calculate the Trowbridge-Reitz isotropic BRDF model with the given
    roughness and the given incident and outgoing directions.
    """
    wh = (wi + wo) / np.linalg.norm(wi + wo)
    cos_theta_i = wi[2]
    cos_theta_o = wo[2]
    cos_theta_h = wh[2]
    cos_theta_hi = np.dot(wi, wh)
    cos_theta_ho = np.dot(wo, wh)
    D = trowbridge_reitz_ndf_iso(alpha, cos_theta_h)
    G = trowbridge_reitz_geom_iso(alpha, cos_theta_hi, cos_theta_ho)
    F = fresnel(cos_theta_i, 1.0, 0.314, 3.8)  # 400nm
    return D * G * F / (4 * cos_theta_i * cos_theta_o)


def beckmann_iso(alpha, wi: np.ndarray, wo: np.ndarray):
    """
    Calculate the Beckmann isotropic BRDF model with the given
    roughness and the given incident and outgoing directions.
    """
    wh = (wi + wo) / np.linalg.norm(wi + wo)
    cos_theta_i = wi[2]
    cos_theta_o = wo[2]
    cos_theta_h = wh[2]
    cos_theta_hi = np.dot(wi, wh)
    cos_theta_ho = np.dot(wo, wh)
    D = beckmann_ndf_iso(alpha, cos_theta_h)
    G = beckmann_geom_iso(alpha, cos_theta_hi, cos_theta_ho)
    F = fresnel(cos_theta_i, 1.0, 0.392, 4.305)
    return D * G * F / (4 * cos_theta_i * cos_theta_o)


def trowbridge_reitz_jacobian(alpha, wi: np.ndarray, wo: np.ndarray):
    """
    Calculate the Jacobian of the Trowbridge-Reitz isotropic BRDF model with respect to the parameters.
    """
    wh = (wi + wo) / np.linalg.norm(wi + wo)
    cos_theta_h = wh[2]
    cos_theta_h2 = cos_theta_h * cos_theta_h
    cos_theta_h4 = cos_theta_h2 * cos_theta_h2
    if cos_theta_h4 < 1e-8:
        return 0.0
    cos_theta_i = wi[2]
    cos_theta_o = wo[2]
    if cos_theta_i < 1e-8 or cos_theta_o < 1e-8:
        return 0.0
    tan_theta_h2 = (1 - cos_theta_h2) / cos_theta_h2
    if tan_theta_h2 == np.inf:
        return 0.0
    f = fresnel(cos_theta_i, 1.0, 0.392, 4.305)
    alpha2 = alpha * alpha
    alpha3 = alpha2 * alpha
    alpha5 = alpha3 * alpha2
    cos_theta_hi = np.abs(np.dot(wi, wh))
    cos_theta_ho = np.abs(np.dot(wo, wh))
    cos_theta_hi2 = cos_theta_hi * cos_theta_hi
    cos_theta_ho2 = cos_theta_ho * cos_theta_ho
    tan_theta_hi2 = (1 - cos_theta_hi2) / cos_theta_hi2
    tan_theta_ho2 = (1 - cos_theta_ho2) / cos_theta_ho2
    ai = np.sqrt(1.0 + alpha2 * tan_theta_hi2)
    ao = np.sqrt(1.0 + alpha2 * tan_theta_ho2)
    one_plus_ai = 1.0 + ai
    one_plus_ao = 1.0 + ao
    one_plus_ai2 = one_plus_ai * one_plus_ai
    one_plus_ao2 = one_plus_ao * one_plus_ao
    sec_theta_i = 1.0 / cos_theta_i
    sec_theta_o = 1.0 / cos_theta_o
    sec_theta_h4 = 1.0 / cos_theta_h4
    nominator = f * sec_theta_h4 * sec_theta_i * sec_theta_o * (
            alpha3 * one_plus_ai * (2.0 + 2.0 * ao + 3.0 * alpha2 * tan_theta_ho2)
            + alpha5 * tan_theta_hi2 * (3.0 + 3.0 * ao + 4.0 * alpha2 * tan_theta_ho2)
            - alpha * tan_theta_h2 * (alpha2 * one_plus_ao * tan_theta_hi2 + one_plus_ai * (
            2.0 + 2.0 * ao + alpha2 * tan_theta_ho2)))
    denominator = np.pi * one_plus_ai2 * one_plus_ao2 * np.power(alpha2 + tan_theta_h2, 3) * ai * ao
    return -nominator / denominator


def residuals_trowbridge_reitz_iso(x, wis, wos, snapshots):
    """
    Calculate the residuals of the BRDF model with the given parameters x
    and the given data.
    """
    # Calculate the BRDF model
    brdf = np.zeros((len(wis), len(wos)))
    for i in range(len(wis)):
        for j in range(len(wos)):
            brdf[i] += trowbridge_reitz_iso(x, wis[i], wos[j])

    # Calculate the residuals
    return (brdf - snapshots).flatten()


def residuals_beckmann_iso(x, wis, wos, snapshots):
    """
    Calculate the residuals of the BRDF model with the given parameters x
    and the given data.
    """
    # Calculate the BRDF model
    brdf = np.zeros((len(wis), len(wos)))
    for i in range(len(wis)):
        for j in range(len(wos)):
            brdf[i] += beckmann_iso(x, wis[i], wos[j])

    # Calculate the residuals
    return (brdf - snapshots).flatten()


def jacobian_trowbridge_reitz_iso(x, wis, wos, _snapshots):
    """
    Calculate the Jacobian of the residuals with respect to the parameters.
    """
    jacobian = np.zeros((len(wis) * len(wos), 1))
    for i in range(len(wis)):
        for j in range(len(wos)):
            jacobian[i * len(wos) + j] = trowbridge_reitz_jacobian(x, wis[i], wos[j])
    return jacobian


def plot_data_surface(wi_idx, wis_sph, wos_cart, snapshots):
    """
    Plot the data as a surface.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = wos_cart[:, 0]
    y = wos_cart[:, 1]
    z = snapshots[wi_idx, :]
    tri = mtri.Triangulation(x, y)
    ax.plot_trisurf(x, y, z, triangles=tri.triangles, cmap='viridis')
    ax.set_title(f'wi: {np.degrees(wis_sph[wi_idx, 0])}, {np.degrees(wis_sph[wi_idx, 1])}')
    plt.show()


x0 = np.array([0.9])

if __name__ == '__main__':
    # Read the data from the file
    wis_sph, wos_sph, snapshots = read_brdf_data(sys.argv[1])

    # Convert the direction from spherical coordinates to Cartesian coordinates
    wis_cart = np.array(
        [np.sin(wis_sph[:, 0]) * np.cos(wis_sph[:, 1]), np.sin(wis_sph[:, 0]) * np.sin(wis_sph[:, 1]),
         np.cos(wis_sph[:, 0])]).T
    wos_cart = np.array(
        [np.sin(wos_sph[:, 0]) * np.cos(wos_sph[:, 1]), np.sin(wos_sph[:, 0]) * np.sin(wos_sph[:, 1]),
         np.cos(wos_sph[:, 0])]).T

    for i in range(len(wis_sph)):
        plot_data_surface(i, wis_sph, wos_cart, snapshots)

    # fun = residuals_beckmann_iso if sys.argv[2] == 'beckmann' else residuals_trowbridge_reitz_iso
    # # Make an initial guess
    # x0 = 1.0
    #
    # # Minimize the function
    # res = opt.least_squares(fun=fun, x0=x0, method='dogbox',
    #                         # jac=jacobian_trowbridge_reitz_iso,
    #                         bounds=(0, 1.5),
    #                         args=(wis_cart, wos_cart, snapshots))
    #
    # # Print the result
    # if res.success:
    #     print(f"Success! {res.x}")
    # else:
    #     print("Failure!")

    print(res)
