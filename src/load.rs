
use rmp_serde::{Deserializer, Serializer};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use itertools::Itertools;
use std::error::Error;

static EXPORT_PATH: &str = "./data_out";

#[derive(Serialize, Deserialize, Debug)]
pub struct Centroids(Vec<Vec<u32>>);

#[derive(Serialize, Deserialize, Debug)]
pub struct Labels(Vec<usize>);

#[derive(Serialize, Deserialize, Debug)]
pub struct Histograms(Vec<Vec<u8>>);

impl Histograms {
    pub fn data(&self) -> &[Vec<u8>] {
        return self.0.as_slice();
    }
}

pub fn save_data(labels: &Vec<usize>, centroids: &Vec<Vec<u32>>, round: usize, initialization_index: usize) -> Result<(), Box<dyn std::error::Error>> {
    let filepath_labels = format!("{}/labels_round_{}_initialization_{}.bin", EXPORT_PATH, round, initialization_index);
    let filepath_centroids = format!("{}/centroids_round_{}_initialization_{}.bin", EXPORT_PATH, round, initialization_index);
    let output_file_labels = File::create(filepath_labels)?;
    let output_file_centroids = File::create(filepath_centroids)?;


    let mut writer_labels = BufWriter::new(output_file_labels);
    let labels = Labels(labels.clone());
    labels.serialize(&mut Serializer::new(&mut writer_labels))?;

    let mut writer_centroids = BufWriter::new(output_file_centroids);
    let centroids = Centroids(centroids.clone());
    centroids.serialize(&mut Serializer::new(&mut writer_centroids))?;

    Ok(())
}

fn convert_dataset(data: &[Vec<u8>]) -> Vec<Vec<u32>> {
    data.iter()
        .map(|vec| vec.iter().map(|&val| val as u32).collect())
        .collect()
}

fn load_data(filepath: &str) -> Result<Vec<Vec<u32>>, Box<dyn std::error::Error>> {
    // Open the file in read-only mode
    let file = File::open(filepath)?;

    // Create a buffered reader for efficient reading
    let buf_reader = BufReader::new(file);

    // Deserialize the data from the reader
    // Deserializer automatically handles the data according to the Histograms structure
    let mut deserializer = Deserializer::new(buf_reader);
    let histograms: Histograms = Deserialize::deserialize(&mut deserializer)?;

    println!("Loaded data: len() = {}", histograms.0.len());
    return Ok(convert_dataset(histograms.data()));
}

pub struct HistogramLoader {
    pub folder_path: String,
    pub filenames: Vec<String>,
    pub round: usize,
    pub histograms: Vec<Vec<u32>>,
}

impl HistogramLoader {
    pub fn new(round: usize) -> Result<Self, Box<dyn Error>> {
        let folder_path = "/Users/kade/git/personal/pluribus/kmeans-kpp-emd-triangle-inequality/data_in".to_string();

        let entries = fs::read_dir(&folder_path)?;
        let all_filenames: Vec<String> = entries.map(|entry| {
            if let Ok(entry) = entry {
                if let Ok(file_name) = entry.file_name().into_string() {
                    return file_name;
                }
            }

            return "".to_string();
        })
            .filter(|file_name| file_name != "")
            .collect_vec();

        let round_filenames: Vec<String> = all_filenames.iter()
            .cloned()
            .filter(|file_name| file_name.starts_with(format!("hsh_round_{}_batch_", round).as_str()))
            .collect();
        println!("round filenames: {:?}", round_filenames);

        let mut histograms: Vec<Vec<u32>> = vec![];
        for (index, round_batch_filename) in round_filenames.iter().enumerate() {
            let filepath = format!("{}/{}", &folder_path, round_batch_filename);
            let batch_histograms = load_data(&filepath)?;
            histograms.extend(batch_histograms);
            println!("Loaded data batch #{}", index+1);
        }

        return Ok(Self {
            folder_path,
            filenames: round_filenames,
            round,
            histograms
        })
    }
}

