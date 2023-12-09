import glob
import json
import os

import re
import regex

import fire

import librosa

import pandas as pd

import pyspark
import pyspark.sql.functions as F
import pyspark.sql.types as T

import spacy

import srt

from schemas import ARCHIVE_ORG_SCHEMA

from io import StringIO
from html.parser import HTMLParser

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = StringIO()
    def handle_data(self, d):
        self.text.write(d)
    def get_data(self):
        return self.text.getvalue()

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def main(nemo_manifest_path="output_nfa_segment_split/part-00000-e10f693c-60c7-41b8-ad07-5ecf61d9d7fb-c000_with_output_file_paths.json",
         input_catalogue_path="CC_BY_SA.jsonl",
         output_file_json="output_prompts/title_prompt.json",
         prompt_type="title"):
    spark = (
        pyspark.sql.SparkSession.builder.master("local[16]")
        .config("spark.eventLog.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config(
            "spark.driver.extraJavaOptions",
            "-Dio.netty.tryReflectionSetAccessible=true",
        )
        .config(
            "spark.executor.extraJavaOptions",
            "-Dio.netty.tryReflectionSetAccessible=true",
        )
        .config("spark.driver.memory", f"32g")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "1")
        .config("spark.local.dir", "spark_local")
        .getOrCreate()
    )

    audio_df = spark.read.format("json").load(nemo_manifest_path)

    catalogue_df = spark.read.format("json").schema(ARCHIVE_ORG_SCHEMA).load(input_catalogue_path)
    exploded_df = catalogue_df.withColumn("exploded_files", F.explode(catalogue_df.files))
    audio_filepath = F.concat(F.lit("scp_commands/"), F.col("identifier"), F.lit("/"), F.col("exploded_files.name"))
    exploded_df = exploded_df.withColumn("audio_filepath", audio_filepath)

    # print("GALVEZ1:", exploded_df.select("audio_filepath").head())
    # print("GALVEZ2:", audio_df.select("audio_filepath").head())
    exploded_df = exploded_df.join(audio_df, "audio_filepath")

    # print("GALVEZ:", exploded_df.join(audio_df, "audio_filepath").count())
    
    metadata_json = "metadata.json_single"
    exploded_df.select(F.col("audio_filepath"), F.col("metadata.description"), F.col("metadata.title"), F.col("metadata.subject")).write.mode("overwrite").format("json").save("metadata.json")
    spark.read.format("json").load("metadata.json").coalesce(1).write.mode("overwrite").format("json").save(metadata_json)

    metadata_json_name = glob.glob(os.path.join(metadata_json, "*.json"))[0]

    nlp = spacy.load("en_core_web_sm")

    filepath_to_metadata = {}
    with open(metadata_json_name) as fh:
        for line in fh:
            dictionary = json.loads(line)
            if prompt_type == "title":
                filepath_to_metadata[dictionary["audio_filepath"]] = dictionary["title"]
            elif prompt_type == "description":
                filepath_to_metadata[dictionary["audio_filepath"]] = dictionary.get("description", "")
            elif prompt_type == "description_no_html":
                filepath_to_metadata[dictionary["audio_filepath"]] = strip_tags(dictionary.get("description", ""))
            elif prompt_type == "nouns_and_proper_nouns":
                nouns = set()

                for document in [nlp(dictionary["title"]), nlp(strip_tags(dictionary.get("description", "")))]:
                    for token in document:
                        if token.pos_ in ("PROPN", "NOUN"):
                            nouns.add(token.text)
                prompt = ",".join(nouns)
                filepath_to_metadata[dictionary["audio_filepath"]] = prompt
            elif prompt_type == "none":
                filepath_to_metadata[dictionary["audio_filepath"]] = ""
            else:
                assert False

    with open(output_file_json, "w") as out_fh:
        json.dump(filepath_to_metadata, out_fh)

    print("GALVEZ:done")

if __name__ == "__main__":
    fire.Fire(main)
