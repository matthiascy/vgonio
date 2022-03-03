use serde::{Deserialize, Serialize};
mod io;

/// Static variable used to generate height field name.
static mut HEIGHT_FIELD_COUNTER: u32 = 0;

/// Helper function to generate a default name for height field instance.
fn gen_height_field_name() -> String {
    unsafe {
        let name = format!("height_field_{:03}", HEIGHT_FIELD_COUNTER);
        HEIGHT_FIELD_COUNTER += 1;
        name
    }
}

/// Alignment used when generating height field.
#[repr(u32)]
#[derive(Debug, Copy, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum AxisAlignment {
    XY,
    XZ,
    YX,
    YZ,
    ZX,
    ZY,
}

impl Default for AxisAlignment {
    fn default() -> Self {
        AxisAlignment::XY
    }
}

/// Representation of the micro-surface.
#[derive(Debug, Serialize, Deserialize)]
pub struct HeightField {
    /// Generated unique identifier.
    pub uuid: uuid::Uuid,
    /// User defined name for the height field.
    pub name: String,
    /// The initial axis alignment in world space. The default is XY, aligned
    /// with the "ground" plane.
    pub orientation: AxisAlignment,
    /// Number of sample points in horizontal direction (first axis of
    /// orientation).
    pub rows: usize,
    /// Number of sample points in vertical direction (second axis of
    /// orientation).
    pub cols: usize,
    /// The space between sample points in horizontal direction.
    pub du: f32,
    /// The space between sample points in vertical direction.
    pub dv: f32,
    /// Height field's center position in world space.
    pub center: [f32; 3],
    /// Minimum height of the height field.
    pub min: f32,
    /// Maximum height of the height field.
    pub max: f32,
    /// Height values of sample points on the height field (values are stored in
    /// row major order).
    pub samples: Vec<f32>,
}

impl HeightField {
    /// Creates a flat height field with specified height value.
    ///
    /// # Arguments
    ///
    /// * `cols` - the number of sample points in dimension x
    /// * `rows` - the number of sample points in dimension y
    /// * `du` - spacing between samples points in dimension x
    /// * `dv` - spacing between samples points in dimension y
    /// * `height` - the initial value of the height
    /// * `orientation` - axis alignment of height field
    ///
    /// # Examples
    ///
    /// ```
    /// # use vgonio::height_field::{AxisAlignment, HeightField};
    /// let height_field = HeightField::new(10, 10, 0.11, 0.11, 0.12, Default::default());
    /// assert_eq!(height_field.samples_count(), 100);
    /// assert_eq!(height_field.cells_count(), 81);
    /// ```
    pub fn new(
        cols: usize,
        rows: usize,
        du: f32,
        dv: f32,
        height: f32,
        orientation: AxisAlignment,
    ) -> Self {
        assert!(cols > 1 && rows > 1);
        let mut samples = Vec::new();
        samples.resize(cols * rows, height);
        HeightField {
            uuid: uuid::Uuid::new_v4(),
            name: gen_height_field_name(),
            orientation,
            rows,
            cols,
            du,
            dv,
            center: [0.0, 0.0, 0.0],
            min: height,
            max: height,
            samples,
        }
    }

    /// Creates a height field and sets its height values by using a function.
    ///
    /// # Arguments
    ///
    /// * `cols` - the number of sample points in dimension x
    /// * `rows` - the number of sample points in dimension y
    /// * `du` - horizontal spacing
    /// * `dv` - vertical spacing
    /// * `orientation` - axis alignment of height field
    /// * `setter` - the setting function, this function will be invoked with
    ///   the row number and column number as parameters.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vgonio::height_field::{AxisAlignment, HeightField};
    /// let height_field = HeightField::new_by(4, 4, 0.1, 0.1, AxisAlignment::XZ, |row, col| (row + col) as f32);
    /// assert_eq!(height_field.samples_count(), 16);
    /// assert_eq!(height_field.max, 6.0);
    /// assert_eq!(height_field.min, 0.0);
    /// assert_eq!(height_field.samples[0], 0.0);
    /// assert_eq!(height_field.samples[2], 2.0);
    /// assert_eq!(height_field.sample_at(2, 3), 5.0);
    /// ```
    pub fn new_by<F>(
        cols: usize,
        rows: usize,
        du: f32,
        dv: f32,
        orientation: AxisAlignment,
        setter: F,
    ) -> HeightField
    where
        F: Fn(usize, usize) -> f32,
    {
        assert!(cols > 1 && rows > 1);
        let mut samples = Vec::with_capacity(cols * rows);
        for r in 0..rows {
            for c in 0..cols {
                samples.push(setter(r, c));
            }
        }
        let max = *samples
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let min = *samples
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        HeightField {
            uuid: uuid::Uuid::new_v4(),
            name: gen_height_field_name(),
            orientation,
            cols,
            rows,
            du,
            dv,
            center: [0.0, 0.0, 0.0],
            max,
            min,
            samples,
        }
    }

    /// Create a height field from a array of elevation values.
    ///
    /// # Arguments
    ///
    /// * `cols` - number of columns (number of sample points in dimension x) in
    ///   the height field.
    /// * `rows` - number of rows (number of sample points in dimension y) in
    ///   the height field.
    /// * `du` - horizontal spacing between two samples
    /// * `dv` - vertical spacing between two samples
    /// * `samples` - array of elevation values of the height field.
    /// * `orientation` - axis alignment of height field
    ///
    /// # Examples
    ///
    /// ```
    /// # use vgonio::height_field::HeightField;
    /// let samples = vec![0.1, 0.2, 0.1, 0.15, 0.11, 0.23, 0.15, 0.1, 0.1];
    /// let height_field = HeightField::from_samples(3, 3, 0.5, 0.5, samples, Default::default());
    /// assert_eq!(height_field.samples_count(), 9);
    /// assert_eq!(height_field.cells_count(), 4);
    /// assert_eq!(height_field.cols, 3);
    /// assert_eq!(height_field.rows, 3);
    /// ```
    pub fn from_samples(
        cols: usize,
        rows: usize,
        du: f32,
        dv: f32,
        samples: Vec<f32>,
        orientation: AxisAlignment,
    ) -> HeightField {
        assert!(cols > 0 && rows > 0 && samples.len() >= cols * rows);
        let max = samples
            .iter()
            .fold(f32::MIN, |acc, x| if *x > acc { *x } else { acc });
        let min = samples
            .iter()
            .fold(f32::MAX, |acc, x| if *x < acc { *x } else { acc });
        HeightField {
            uuid: uuid::Uuid::new_v4(),
            name: gen_height_field_name(),
            orientation,
            rows,
            cols,
            du,
            dv,
            center: [0.0, 0.0, 0.0],
            max,
            min,
            samples,
        }
    }

    /// Returns the dimension of the surface [rows * du, cols * dv]
    /// # Examples
    ///
    /// ```
    /// # use vgonio::height_field::HeightField;
    /// let height_field = HeightField::new(100, 100, 0.1, 0.1, 0.1, Default::default());
    /// assert_eq!(height_field.dimension(), (10.0, 10.0));
    /// ```
    pub fn dimension(&self) -> (f32, f32) {
        (self.rows as f32 * self.du, self.cols as f32 * self.dv)
    }

    /// Returns the number of samples of height field.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vgonio::height_field::HeightField;
    /// let samples = vec![0.1, 0.2, 0.1, 0.15, 0.11, 0.23, 0.15, 0.1, 0.1];
    /// let height_field = HeightField::from_samples(3, 3, 0.2, 0.2, samples, Default::default());
    /// assert_eq!(height_field.samples_count(), 9);
    /// ```
    pub fn samples_count(&self) -> usize {
        self.cols * self.rows
    }

    /// Returns the number of cells of height field.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vgonio::height_field::HeightField;
    /// let samples = vec![0.1, 0.2, 0.1, 0.15, 0.11, 0.23, 0.15, 0.1, 0.1];
    /// let height_field = HeightField::from_samples(3, 3, 0.2, 0.2, samples, Default::default());
    /// assert_eq!(height_field.cells_count(), 4);
    /// ```
    pub fn cells_count(&self) -> usize {
        if self.cols == 0 || self.rows == 0 {
            0
        } else {
            (self.cols - 1) * (self.rows - 1)
        }
    }

    /// Returns the height value of a given sample.
    ///
    /// # Arguments
    ///
    /// * `col` - sample's position in dimension x
    /// * `row` - sample's position in dimension y
    ///
    /// # Examples
    ///
    /// ```
    /// # use vgonio::height_field::HeightField;
    /// let samples = vec![0.1, 0.2, 0.1, 0.15, 0.11, 0.23, 0.15, 0.1, 0.1];
    /// let height_field = HeightField::from_samples(3, 3, 0.2, 0.2, samples, Default::default());
    /// assert_eq!(height_field.sample_at(2, 2), 0.1);
    /// ```
    ///
    /// ```should_panic
    /// # use vgonio::height_field::HeightField;
    /// let samples = vec![0.1, 0.2, 0.1, 0.15, 0.11, 0.23, 0.15, 0.1, 0.1];
    /// let height_field = HeightField::from_samples(3, 3, 0.2, 0.3, samples, Default::default());
    /// let h = height_field.sample_at(4, 4);
    /// ```
    pub fn sample_at(&self, col: usize, row: usize) -> f32 {
        assert!(col < self.cols);
        assert!(row < self.rows);
        self.samples[col * self.rows + row]
    }
}
