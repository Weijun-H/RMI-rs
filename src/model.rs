pub trait Model {
    fn predict(&self, x: f64) -> f64;
    fn fit(&mut self, x: &[f64], y: &[f64]);
    fn params(&self) -> Vec<f64>;
}
pub struct LinearRegression {
    pub intercept: f64,
    pub slope: f64,
}

impl Model for LinearRegression {
    fn predict(&self, x: f64) -> f64 {
        self.intercept + self.slope * x
    }
    fn fit(&mut self, x: &[f64], y: &[f64]) {
        let n = x.len() as f64;
        let sum_x = x.iter().sum::<f64>();
        let sum_y = y.iter().sum::<f64>();
        let sum_xy = x.iter().zip(y.iter()).map(|(&x, &y)| x * y).sum::<f64>();
        let sum_xx = x.iter().map(|&x| x * x).sum::<f64>();
        self.slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
        self.intercept = (sum_y - self.slope * sum_x) / n;
    }
    fn params(&self) -> Vec<f64> {
        vec![self.intercept, self.slope]
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_linear_regression() {
        let mut model = LinearRegression {
            intercept: 0.0,
            slope: 0.0,
        };
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];
        model.fit(&x, &y);
        assert_eq!(model.predict(4.0), 4.0);
        assert_eq!(model.params()[0], 0.0);
        assert_eq!(model.params()[1], 1.0);
    }
}
