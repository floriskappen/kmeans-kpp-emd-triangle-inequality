
use chrono::round;
use prost::Message;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use itertools::Itertools;
use std::error::Error;

use crate::proto::{ClusteredDataCentroids, ClusteredDataLabels, FloatList, HandStrengthHistograms};

static EXPORT_PATH: &str = "./data_out";

pub fn save_data(labels: Vec<u32>, centroids: Vec<Vec<f32>>, round: usize, initialization_index: usize) -> Result<(), Box<dyn std::error::Error>> {
    let filepath_labels = format!("{}/labels_round_{}_initialization_{}.bin", EXPORT_PATH, round, initialization_index);
    let labels = ClusteredDataLabels {
        data: labels
    };
    let mut labels_buf = Vec::new();
    labels.encode(&mut labels_buf).expect("Error encoding labels");
    drop(labels);
    let mut labels_file = BufWriter::new(File::create(filepath_labels)?);
    labels_file.write_all(&labels_buf).expect("Error writing labels to file");


    let filepath_centroids = format!("{}/centroids_round_{}_initialization_{}.bin", EXPORT_PATH, round, initialization_index);
    // let centroids = centroids.iter().map(|centroid| centroid.iter().map(|&value| value as u8).collect_vec()).collect_vec();
    let centroids = ClusteredDataCentroids {
        data: centroids.iter().map(|centroid| FloatList { values: centroid.clone() }).collect_vec()
    };
    let mut centroids_buf = Vec::new();
    centroids.encode(&mut centroids_buf).expect("Error encoding centroids");
    let mut centroids_file = BufWriter::new(File::create(filepath_centroids)?);
    centroids_file.write_all(&centroids_buf).expect("Error writing centroids to file");

    Ok(())
}

fn load_data(filepath: &str) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
    let mut buf_reader = BufReader::new(File::open(filepath)?);
    let mut buf = Vec::new();
    buf_reader.read_to_end(&mut buf)?;

    let hand_strenght_histograms = HandStrengthHistograms::decode(&*buf)?;
    println!("Loaded data: len() = {}", hand_strenght_histograms.data.len());
    return Ok(hand_strenght_histograms.data);
}

pub struct HistogramLoader {
    pub folder_path: String,
    pub filenames: Vec<String>,
    pub round: usize,
    pub histograms: Vec<Vec<u8>>,
}

impl HistogramLoader {
    pub fn new(round: usize) -> Result<Self, Box<dyn Error>> {
        let folder_path = "./data_in".to_string();

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

        let mut round_filenames: Vec<String> = all_filenames.iter()
            .cloned()
            .filter(|file_name| file_name.starts_with(format!("round_{}_batch_", round).as_str()))
            .collect();
        round_filenames.sort_by_key(|filename| {
            filename
                .split('_')
                .nth(3)  // This gets the part of the filename with the batch number
                .and_then(|s| s.split('.').next())  // Remove the file extension
                .and_then(|num| num.parse::<i32>().ok())  // Parse the number part as i32
                .unwrap_or(0)  // Default to 0 if any parsing fails
        });
        println!("round filenames: {:?}", round_filenames);

        // Pre-calculate the total number of histogram entries
        let total_entries = round_filenames.iter()
            .map(|filename| {
                let filepath = format!("{}/{}", &folder_path, filename);
                let mut buf_reader = BufReader::new(File::open(&filepath).unwrap());
                let mut buf = Vec::new();
                buf_reader.read_to_end(&mut buf).unwrap();
                let histograms = HandStrengthHistograms::decode(&*buf).unwrap();
                histograms.data.len()
            })
            .sum::<usize>();

        log::info!("scanned files to find {} entries", total_entries);

        // Allocate memory efficiently knowing each histogram is of size 8 bytes
        let mut histograms: Vec<Vec<u8>> = Vec::with_capacity(total_entries);
        for filename in &round_filenames {
            let filepath = format!("{}/{}", &folder_path, filename);
            let mut file_histograms = load_data(&filepath)?;
            
            // Ensure each histogram uses exactly the memory it needs, set to 8 bytes
            file_histograms.iter_mut().for_each(|histogram| {
                let mut optimized_histogram = Vec::with_capacity(8);
                optimized_histogram.extend_from_slice(histogram);
                histograms.push(optimized_histogram);
            });

            log::info!("loaded file {}", filename);
        }

        log::info!("loaded data from {} files to memory", round_filenames.len());

        return Ok(Self {
            folder_path,
            filenames: round_filenames,
            round,
            histograms
        })
    }
}
