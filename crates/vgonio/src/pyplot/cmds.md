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