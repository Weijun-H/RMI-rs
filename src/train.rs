use crate::model::*;

pub fn train<M: Model>(model: &mut M, x: &[f64], y: &[f64]) {
    model.fit(x, y);
}

pub fn predict<M: Model>(model: &M, x: f64) -> f64 {
    model.predict(x)
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
        train(&mut model, &x, &y);
        assert_eq!(predict(&model, 4.0), 4.0);
        assert_eq!(model.params()[0], 0.0);
        assert_eq!(model.params()[1], 1.0);
    }
}
