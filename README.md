# vgonio

## Requirements

* [shaderc](https://github.com/google/shaderc)
* [Embree3](https://www.embree.org/)(optional)

## Documentation

To support latex, when generating the documentation extra html header content will be added. With `cargo doc` or 
`cargo rustdoc`, the `rustdocflags` defined in `./cargo/config.toml` will be used. With `cargo doc`, `rustdocflags` is
applied globally (applies to each dependent crate), it will mess up dependencies. Possible workarounds are to either add
`--no-deps` to the `cargo doc`, or use `cargo rustdoc` to generate documentation only for the root crate(this crate). 
For more information, see cargo issue [#331](https://github.com/rust-lang/cargo/issues/331)
and rust pull request [#95691](https://github.com/rust-lang/rust/pull/95691).

## File formats

## Cache file (.vgc)

### Header (binary format)

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">values(hex)</th>
<th scope="col" class="org-left">purpose</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">4 bytes</td>
<td class="org-left">in ASCII, &ldquo;DCCC&rdquo; &ldquo;0x44 0x43 0x43 0x43&rdquo;</td>
</tr>


<tr>
<td class="org-left">1 byte</td>
<td class="org-left">&ldquo;!(0x33)&rdquo; for binary content, &ldquo;#(0x35)&rdquo; for plain text content</td>
</tr>


<tr>
<td class="org-left">1 byte</td>
<td class="org-left">cache type</td>
</tr>


<tr>
<td class="org-left">4 bytes</td>
<td class="org-left">content size in bytes</td>
</tr>
</tbody>
</table>

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">cache type</th>
<th scope="col" class="org-right">value</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">bvh</td>
<td class="org-right">0x01</td>
</tr>


<tr>
<td class="org-left">height field</td>
<td class="org-right">0x02</td>
</tr>


<tr>
<td class="org-left">mesh</td>
<td class="org-right">0x04</td>
</tr>
</tbody>
</table>


<a id="org36277b9"></a>

### Body

Serialized either by a yaml serializer (in ascii format) or by a bincode serializer (in binary format).


<a id="orgf1c88e4"></a>

## Micro-surface file (.vgm)

All of the values are stored in **little-endian** format.


<a id="orga97b87b"></a>

### Header (binary format)

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">values (hex)</th>
<th scope="col" class="org-left">purpose</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">4 bytes</td>
<td class="org-left">in ASCII, &ldquo;DCMS&rdquo; &ldquo;44 43 4D 53&rdquo;</td>
</tr>


<tr>
<td class="org-left">1 byte</td>
<td class="org-left">binary &ldquo;!(0x33)&rdquo; or plain text &ldquo;#(0x35)&rdquo;</td>
</tr>


<tr>
<td class="org-left">1 bytes</td>
<td class="org-left">SI unit, could be micrometre &ldquo;0x01&rdquo; /nanometre &ldquo;0x02&rdquo;</td>
</tr>


<tr>
<td class="org-left">4 bytes</td>
<td class="org-left">Horizontal spacing between two sample points</td>
</tr>


<tr>
<td class="org-left">4 bytes</td>
<td class="org-left">Vertical spacing between two sample points</td>
</tr>


<tr>
<td class="org-left">4 bytes</td>
<td class="org-left">Microsurface horizontal samples count</td>
</tr>


<tr>
<td class="org-left">4 bytes</td>
<td class="org-left">Microsurface vertical samples count</td>
</tr>


<tr>
<td class="org-left">4 bytes</td>
<td class="org-left">The size of the microsurface file in bytes</td>
</tr>


<tr>
<td class="org-left">1 byte</td>
<td class="org-left">newline character 0x0a</td>
</tr>
</tbody>
</table>


<a id="org884ffec"></a>

### Body data format (ascii or binary format)

1.  Binary format
    Sample points of the micro-surface height field are stored continuously as an
    1D array, line by line.

2.  Plain text format
    The data are stored as 2D matrix in a ascii format. One scanline per text line
    from left to right and top to bottom. The x dimension increases along a scanline
    and the y dimension increase with each successive scanline.

