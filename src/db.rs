use anyhow::Context;
use axum::Extension;
use lazy_static::lazy_static;
use rayon::prelude::*;
use schemars::JsonSchema;
use std::{
	collections::{BinaryHeap, HashMap},
	fs,
	path::PathBuf,
	sync::Arc,
};
use tokio::sync::RwLock;

use crate::similarity::{get_cache_attr, get_distance_fn, normalize, Distance, ScoreIndex};

lazy_static! {
	pub static ref STORE_PATH: PathBuf = PathBuf::from("./storage/db");
}

#[allow(clippy::module_name_repetitions)]
pub type DbExtension = Extension<Arc<RwLock<Db>>>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
	#[error("Collection already exists")]
	UniqueViolation,

	#[error("Collection doesn't exist")]
	NotFound,

	#[error("The dimension of the vector doesn't match the dimension of the collection")]
	DimensionMismatch,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct Db {
	pub collections: HashMap<String, Collection>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
pub struct SimilarityResult {
	score: f32,
	embedding: Embedding,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
pub struct Collection {
	/// Dimension of the vectors in the collection
	pub dimension: usize,
	/// Distance metric used for querying
	pub distance: Distance,
	/// Embeddings in the collection
	#[serde(default)]
	pub embeddings: Vec<Embedding>,
}

impl Collection {
	pub fn get_similarity(&self, query: &[f32], k: usize) -> Vec<SimilarityResult> {
		let query_vec = if self.distance == Distance::Cosine {
			normalize(query)
		} else {
			query.to_vec()
		};
		let memo_attr = get_cache_attr(self.distance, &query_vec);
		let distance_fn = get_distance_fn(self.distance);

		let scores = self
			.embeddings
			.par_iter()
			.enumerate()
			.map(|(index, embedding)| {
				let score = distance_fn(&embedding.vector, &query_vec, memo_attr);
				ScoreIndex { score, index }
			})
			.collect::<Vec<_>>();

		let mut heap = BinaryHeap::new();
		for score_index in scores {
			if heap.len() < k || score_index < *heap.peek().unwrap() {
				heap.push(score_index);

				if heap.len() > k {
					heap.pop();
				}
			}
		}

		heap.into_sorted_vec()
			.into_iter()
			.map(|ScoreIndex { score, index }| SimilarityResult {
				score,
				embedding: self.embeddings[index].clone(),
			})
			.collect()
	}
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
pub struct Embedding {
	pub id: String,
	pub vector: Vec<f32>,
	pub metadata: Option<HashMap<String, String>>,
}

impl Db {
	pub fn new() -> Self {
		Self {
			collections: HashMap::new(),
		}
	}

	pub fn extension(self) -> DbExtension {
		Extension(Arc::new(RwLock::new(self)))
	}

	pub fn create_collection(
		&mut self,
		name: String,
		dimension: usize,
		distance: Distance,
	) -> Result<Collection, Error> {
		if self.collections.contains_key(&name) {
			return Err(Error::UniqueViolation);
		}

		let collection = Collection {
			dimension,
			distance,
			embeddings: Vec::new(),
		};

		self.collections.insert(name, collection.clone());

		Ok(collection)
	}

	pub fn delete_collection(&mut self, name: &str) -> Result<(), Error> {
		if !self.collections.contains_key(name) {
			return Err(Error::NotFound);
		}

		self.collections.remove(name);

		Ok(())
	}

	pub fn insert_into_collection(
		&mut self,
		collection_name: &str,
		mut embedding: Embedding,
	) -> Result<(), Error> {
		let collection = self
			.collections
			.get_mut(collection_name)
			.ok_or(Error::NotFound)?;

		if collection.embeddings.iter().any(|e| e.id == embedding.id) {
			return Err(Error::UniqueViolation);
		}

		if embedding.vector.len() != collection.dimension {
			return Err(Error::DimensionMismatch);
		}

		if collection.distance == Distance::Cosine {
			embedding.vector = normalize(&embedding.vector);
		}

		collection.embeddings.push(embedding);

		Ok(())
	}

	pub fn get_collection(&self, name: &str) -> Option<&Collection> {
		self.collections.get(name)
	}

	fn load_from_store() -> anyhow::Result<Self> {
		if !STORE_PATH.exists() {
			tracing::debug!("Creating database store");
			fs::create_dir_all(STORE_PATH.parent().context("Invalid store path")?)?;

			return Ok(Self::new());
		}

		tracing::debug!("Loading database from store");
		let db = fs::read(STORE_PATH.as_path())?;
		Ok(bincode::deserialize(&db[..])?)
	}

	fn save_to_store(&self) -> anyhow::Result<()> {
		let db = bincode::serialize(self)?;

		fs::write(STORE_PATH.as_path(), db)?;

		Ok(())
	}
}

impl Drop for Db {
	fn drop(&mut self) {
		tracing::debug!("Saving database to store");
		self.save_to_store().ok();
	}
}

pub fn from_store() -> anyhow::Result<Db> {
	Db::load_from_store()
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::similarity::Distance;

	#[test]
	fn test_cosine_query_scores_reflect_magnitude() {
		let mut db = Db::new();
		db.create_collection("test".to_string(), 3, Distance::Cosine).unwrap();

		db.insert_into_collection("test", Embedding {
			id: "a".to_string(),
			vector: vec![1.0, 0.0, 0.0],
			metadata: None,
		}).unwrap();

		let results = db.get_collection("test").unwrap()
			.get_similarity(&[3.0, 0.0, 0.0], 1);

		assert!(results[0].score > 1.5,
			"Score should reflect query magnitude (expected ~3.0), got {}",
			results[0].score);
	}

	#[test]
	fn test_cosine_ranking_preserves_query_scale() {
		let mut db = Db::new();
		db.create_collection("test".to_string(), 2, Distance::Cosine).unwrap();

		db.insert_into_collection("test", Embedding {
			id: "aligned".to_string(),
			vector: vec![1.0, 0.0],
			metadata: None,
		}).unwrap();
		db.insert_into_collection("test", Embedding {
			id: "diagonal".to_string(),
			vector: vec![1.0, 1.0],
			metadata: None,
		}).unwrap();

		// Query with a scaled vector - ranking should be same regardless of scale
		let results_small = db.get_collection("test").unwrap()
			.get_similarity(&[1.0, 0.0], 2);
		let results_large = db.get_collection("test").unwrap()
			.get_similarity(&[100.0, 0.0], 2);

		// Both queries point in the same direction, so ranking should be identical
		assert_eq!(results_small[0].embedding.id, results_large[0].embedding.id,
			"Ranking should be scale-invariant for cosine");

		// But scores should differ by the scale factor
		let ratio = results_large[0].score / results_small[0].score;
		assert!(ratio > 50.0,
			"Score ratio should reflect query scale (~100x), got {:.1}x", ratio);
	}

	#[test]
	fn test_cosine_ranking_correct_order() {
		let mut db = Db::new();
		db.create_collection("test".to_string(), 2, Distance::Cosine).unwrap();
		db.insert_into_collection("test", Embedding {
			id: "aligned".to_string(),
			vector: vec![1.0, 0.0],
			metadata: None,
		}).unwrap();
		db.insert_into_collection("test", Embedding {
			id: "diagonal".to_string(),
			vector: vec![1.0, 1.0],
			metadata: None,
		}).unwrap();
		let results = db.get_collection("test").unwrap()
			.get_similarity(&[1.0, 0.0], 2);
		assert_eq!(results[0].embedding.id, "aligned",
			"Perfectly aligned vector should rank first for cosine, got '{}'",
			results[0].embedding.id);
	}

	#[test]
	fn test_dot_product_not_affected() {
		let mut db = Db::new();
		db.create_collection("test".to_string(), 2, Distance::DotProduct).unwrap();

		db.insert_into_collection("test", Embedding {
			id: "a".to_string(),
			vector: vec![2.0, 3.0],
			metadata: None,
		}).unwrap();

		let results = db.get_collection("test").unwrap()
			.get_similarity(&[1.0, 1.0], 1);

		assert!((results[0].score - 5.0).abs() < 0.01,
			"DotProduct score should be 5.0, got {}", results[0].score);
	}
}
