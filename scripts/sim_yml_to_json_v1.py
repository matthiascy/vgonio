#!/usr/bin/env python

# Convert old simulation file (yaml) to json file

import os
import sys
import yaml
import re
import json
from math import degrees


def read_sim_yaml(filename):
    with open(filename, 'r') as f:
        return yaml.safe_load(f)


def extract_angle(stat, angle):
    return degrees(stat[angle])


# in the yaml file, the angle is in radians, phi is the polar angle, theta is the azimuthal angle

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: {} <simulation_files>".format(os.path.basename(sys.argv[0])),
              file=sys.stderr)
        exit(-1)

    for filename in sys.argv[1:]:
        sim = read_sim_yaml(filename)

        output = [
            {
                'bounces': [*range(1, 17)],
                'raysCount': sim['params']['rays_count'],
                'wavelength': sim['stats'][0]['wavelength']['Nanometre'],
            }
        ]

        # extract bounces histogram
        for stat in sim['stats']:
            azimuth_i = extract_angle(stat, 'theta_i')
            azimuth_o = extract_angle(stat, 'theta_s')
            polar_i = round(extract_angle(stat, 'phi_i'))
            polar_o = round(extract_angle(stat, 'phi_s'))
            histogram = dict(sorted(stat['reflection_histogram'].items()))
            output.append({
                'phiIn': azimuth_i,
                'phiOut': azimuth_o,
                'thetaIn': polar_i,
                'thetaOut': polar_o,
                'histogram': [histogram[i] if i in histogram else 0 for i in range(1, 17)]
            })

        file_name = os.path.splitext(filename)[0] + '.json'
        abs_path = os.path.abspath(filename)
        dir_path = os.path.dirname(abs_path) + '/sim_yml_to_json'

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        output_filename = os.path.join(dir_path, file_name)

        with open(output_filename, 'w') as f:
            print("Writing {}".format(output_filename))
            f.write(re.sub(r'(\n\s+)(\d+,*|])', r'\g<2>', json.dumps(output, indent=4)))
