//! Helpers shared by the unit tests (compiled only under `cfg(test)`).

/// Tiny 64-bit linear congruential generator with a fixed seed, so tests are
/// reproducible without a `rand` dependency.
pub struct Lcg(u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
    }

    pub fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }

    /// Uniform in `[0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        f64::from(self.next_u32()) / f64::from(1u32 << 31)
    }
}

/// Dense `n_rows x n_cols` 0/1 matrix where each entry is 1 with
/// probability `density`.
pub fn random_binary(seed: u64, n_rows: usize, n_cols: usize, density: f64) -> Vec<Vec<f64>> {
    let mut rng = Lcg::new(seed);
    (0..n_rows)
        .map(|_| {
            (0..n_cols)
                .map(|_| if rng.uniform() < density { 1.0 } else { 0.0 })
                .collect()
        })
        .collect()
}

/// CSR arrays `(data, indices, indptr)` of a dense matrix, zeros dropped.
pub fn csr_from_dense(x: &[Vec<f64>]) -> (Vec<f64>, Vec<u32>, Vec<usize>) {
    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = vec![0usize];
    for row in x {
        for (j, &v) in row.iter().enumerate() {
            if v != 0.0 {
                data.push(v);
                indices.push(j as u32);
            }
        }
        indptr.push(data.len());
    }
    (data, indices, indptr)
}
