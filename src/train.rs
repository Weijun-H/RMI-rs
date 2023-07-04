use std::cmp::{max, min};

use crate::model::*;

pub fn train<M: Model + Clone>(model: &mut M, x: &[f64], y: &[f64]) -> M {
    model.fit(x, y);
    model.clone()
}

// TODO: Extend to n layers
/// Train a model with two layers
/// * `layer` - The number of layers
/// * `l` - The number of models in each layer
/// * `model` - The model to train
/// * `x` - The input data
/// * `y` - The output data
pub fn train_with_two_layers<M: Model + Clone>(
    _layers: usize,
    l: Vec<usize>,
    model: M,
    x: &[f64],
    y: &[f64],
) -> Vec<Vec<M>> {
    // TODO: Only support two layers now
    let layers = 2;
    // Initialize dynamic array to store models
    let mut trained_models: Vec<Vec<M>> = vec![vec![]; layers];
    // Initialize dynamic array to store each model’s keys
    let mut keys: Vec<Vec<(Vec<f64>, Vec<f64>)>> = vec![vec![]; layers];
    for i in 0..layers {
        keys[i] = vec![(vec![], vec![]); l[i]];
    }

    // Assign all keys to the root model
    keys[0][0] = (x.to_vec(), y.to_vec());

    let size = x.len();
    for i in 0..layers {
        for j in 0..l[i] {
            let model = train(&mut model.clone(), &keys[i][j].0, &keys[i][j].1);
            trained_models[i].push(model);
            // Check whether current layer is not last layer
            if i < layers - 1 {
                for (x, y) in keys[i][j].clone().0.iter().zip(keys[i][j].clone().1.iter()) {
                    let p = get_model_index(*x, &trained_models[i][j], l[i + 1], size);
                    keys[i + 1][p].0.push(*x);
                    keys[i + 1][p].1.push(*y);
                }
            }
        }
    }
    trained_models
}

/// Get the model index in the next layer
/// * `key` - The key to predict
/// * `model` - The model to predict
/// * `layer` - The number of layers
/// * `size` - The size of input data
pub fn get_model_index<M: Model>(key: f64, model: &M, layer: usize, size: usize) -> usize {
    max(
        0,
        min(layer * model.predict(key) as usize / size, layer - 1),
    )
}

pub fn predict<M: Model>(trained_models: Vec<Vec<M>>, key: f64, size: usize) -> Option<f64> {
    let mut j = 0;
    for i in 0..trained_models.len() {
        let model = &trained_models[i][j];
        if i == trained_models.len() - 1 {
            return Some(trained_models[i][j].predict(key));
        } else {
            j = get_model_index(key, model, trained_models[i + 1].len(), size);
        };
    }
    None
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
        assert_eq!(model.params()[0], 0.0);
        assert_eq!(model.params()[1], 1.0);
    }

    #[test]
    fn test_train_with_two_layers() {
        let model = LinearRegression {
            intercept: 0.0,
            slope: 0.0,
        };
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];
        let trained_models = train_with_two_layers(2, vec![1, 2], model, &x, &y);
        assert_eq!(predict(trained_models, 3.0, 3), Some(3.0));
    }

    #[test]
    fn test_linear_regression_accuracy() {
        let model = LinearRegression {
            intercept: 0.0,
            slope: 0.0,
        };
        // generate f64 random numbers between 1 and 100
        let x: Vec<f64> = (0..1_000_000)
            .map(|_| rand::random::<f64>() * 100.0)
            .collect();
        let y = x.clone();
        let trained_models = train_with_two_layers(2, vec![1, 10], model, &x, &y);

        let mut count = 0;
        for key in x.clone() {
            let y = predict(trained_models.clone(), key, x.len());
            if y == Some(key) {
                count += 1;
            }
        }
        println!("Total: {}", x.len());
        println!("Accuracy: {}%", count as f64 / x.len() as f64);
    }
}
