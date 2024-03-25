use crate::app::Config;
use base::error::VgonioError;
use exr::{image::Image, prelude::AttributeValue::Text};
use std::{borrow::Cow, path::PathBuf};

/// Options for the `diff` subcommand.
#[derive(clap::Args, Debug, Clone)]
#[clap(about = "Diff two measured data sets.")]
pub struct DiffOptions {
    #[clap(
        help = "Input files to compute the difference between. The first file is considered as \
                the reference."
    )]
    pub inputs: Vec<PathBuf>,
    #[clap(
        long,
        short,
        help = "Output file to save the difference data. If not specified, the difference data \
                will be written to the standard output."
    )]
    pub output: Option<PathBuf>,
}

pub fn diff(opts: DiffOptions, config: Config) -> Result<(), VgonioError> {
    use exr::prelude::*;
    let images = opts
        .inputs
        .iter()
        .map(|p| read_all_data_from_file(p).ok())
        .collect::<Option<Vec<_>>>()
        .unwrap();
    let sizes = images
        .iter()
        .map(|i| i.attributes.display_window.size)
        .collect::<Vec<_>>();
    if !sizes.iter().all(|s| *s == sizes[0]) {
        return Err(VgonioError::new(
            "The two images have different sizes.",
            None,
        ));
    }
    let channels = images
        .iter()
        .map(|i| &i.layer_data[0].channel_data.list[0])
        .collect::<Vec<_>>();
    let image_names = opts
        .inputs
        .iter()
        .map(|p| p.file_stem().unwrap().to_str())
        .collect::<Option<Vec<_>>>()
        .unwrap();
    let layers = (1..images.len())
        .into_iter()
        .map(|i| {
            let diff = channels[0]
                .sample_data
                .get_level(Vec2(0, 0))
                .unwrap()
                .values()
                .zip(
                    channels[i]
                        .sample_data
                        .get_level(Vec2(0, 0))
                        .unwrap()
                        .values(),
                )
                .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
                .collect::<Vec<f32>>();
            let layer_attrib = LayerAttributes::named(Text::from(
                format!("diff-{}-{}", &image_names[0], &image_names[i]).as_str(),
            ));
            let channel =
                std::iter::once(AnyChannel::new("diff", FlatSamples::F32(Cow::Owned(diff))));
            Layer::new(
                sizes[0],
                layer_attrib,
                Encoding::FAST_LOSSLESS,
                AnyChannels {
                    list: SmallVec::from_iter(channel),
                },
            )
        })
        .collect::<Vec<_>>();

    let out_path = opts
        .output
        .unwrap_or_else(|| config.output_dir().to_path_buf())
        .join("diff.exr");
    let diff_image = Image::from_layers(images[0].attributes.clone(), layers);
    diff_image.write().to_file(out_path).map_err(|err| {
        VgonioError::new(
            &format!("Failed to write the difference data to file: {}", err),
            Some(Box::new(err)),
        )
    })?;

    Ok(())
}
