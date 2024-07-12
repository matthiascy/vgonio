# Commands for plotting

## NDF Plot

Single embedded NDF plot with colorbar.

```bash
python pyplot/tone_mapping.py --input ndf_128x128_tl_br.exr --channel NDF --cbar --save ndf_128x128.pdf
```

Single NDF line plot.

```bash
vgonio plot -i ndf_128x128_tl_br.vgmo -k ndf --pi 0
```

NDF difference plot.

```bash
python pyplot/tone_mapping.py --input ndf_al032_tl_br_part.exr  ndf_al032_bl_tr_part.exr --channel NDF --cbar --coord --save ndf_al032_diff_part.pdf --cmap plasma --fc w --diff
```

BRDF Compare Plot

```bash
vgonio plot -i bsdf_aluminium65bar100_2024-06-12T00-29-27.vgmo Aluminium_65bar_100mm_brdf.json --kind cmp-vc --dense --level l0
```