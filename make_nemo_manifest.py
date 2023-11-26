import re
import regex

import fire

import pandas as pd
import pyspark
import pyspark.sql.functions as F
import pyspark.sql.types as T

import librosa

import srt


from schemas import ARCHIVE_ORG_SCHEMA

def main(archive_org_path="scp_commands/",
         input_catalogue_path="CC_BY_SA.jsonl",
         rev_srt_dir="../rev_transcripts/"):
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

    audio_paths = F.concat(
        F.lit(archive_org_path),
        F.lit("/"),
        F.col("identifier"),
        F.lit("/"),
        F.col("audio_document_id"),
    )

    df = spark.read.format("json").schema(ARCHIVE_ORG_SCHEMA).load(input_catalogue_path)

    exploded_df = df.withColumn("exploded_files", F.explode(df.files))

    filtered_exploded_df = exploded_df.where(
        # When a file's size is 0 bytes, scripts/archive.org/download_items.py does
        # not download that file. We therefore filter out size 0 bytes to prevent
        # file-not-found errors in align_lib.py::load_transcripts()
        # https://archive.org/metadata/house.hbs.mars.hrs06RES2154_090401
        (exploded_df.exploded_files.size.cast(T.LongType()) != 0)
        &
        # This indicates that the file is not "private".
        # As far as I can tell, the "private" field is either "true" or null.
        # Trying to read this data as booleans turns every field null for some
        # reason, so it is currently a String field.
        # Private data is not downloadable by the general public.
        exploded_df.exploded_files.private.isNull()
        &
        # "[" and "]" are wild card characters. GCS has very poor support
        # for these. Namely, you can write them but not read them back. More
        # resources here: https://github.com/galv/lingvo-copy/issues/18
        # I simply filter out any files containing these characters for now.
        (
            ~(
                (exploded_df.identifier.contains("["))
                | (exploded_df.identifier.contains("]"))
            )
        )
        & (
            ~(
                (exploded_df.exploded_files["name"].contains("["))
                | (exploded_df.exploded_files["name"].contains("]"))
            )
        )
    )
    audio_df = filtered_exploded_df.select(
        filtered_exploded_df.identifier,
        filtered_exploded_df.exploded_files["name"].alias("audio_document_id"),
    ).where(
        (
            (filtered_exploded_df.exploded_files.format == "MP3")
            | (filtered_exploded_df.exploded_files.format == "VBR MP3")
        )
        &
        # Some non-mp3 files are given the format "MP3". See here:
        # https://ia802901.us.archive.org/4/items/disneychannelourhourafterhourafterhourprankstermarathonapril12004/disneychannelourhourafterhourafterhourprankstermarathonapril12004_files.xml
        (filtered_exploded_df.exploded_files["name"].endswith(".mp3"))
    )

    key_name = F.concat(F.col("identifier"), F.lit("/"), F.col("audio_document_id"))
    key_name = key_name.substr(F.lit(0), F.length(key_name) - 4)
    key_name = F.regexp_replace(key_name, r"/", "_")
    # key_name = F.regexp_replace(key_name, r"\.", "_")
    key_name = F.regexp_replace(key_name, r" ", "_")
    key_name = F.concat(key_name, F.lit(".srt"))

    audio_df = audio_df.withColumn("srt_name", key_name)

    # audio_df.select(key_name).write.mode("overwrite").format("text").save("outputs/keys")

    rev_srts_df = spark.read.format("binaryFile").option("pathGlobFilter", "*.srt").load(rev_srt_dir).drop("modificationTime", "length")
    rev_srts_df = rev_srts_df.select(rev_srts_df.content, F.regexp_replace(F.reverse(F.split(rev_srts_df.path, "/"))[0], r"archive_org_Mar_7_2021_EXPANDED_LICENSES_FILTERED_ACCESS_", "").alias("srt_name"
))
    # We are getting 128 out of 137 matching
    # print("GALVEZ:", len(rev_srts_df.join(audio_df, "srt_name").collect()))
    # print("GALVEZ:", rev_srts_df.select(F.regexp_replace(F.reverse(F.split(rev_srts_df.path, "/"))[0], r"archive_org_Mar_7_2021_EXPANDED_LICENSES_FILTERED_ACCESS_", "").alias("srt_name")).collect())

    manifest_df = rev_srts_df.join(audio_df, "srt_name")
    audio_file_path = F.concat(F.lit(archive_org_path), F.col("identifier"), F.lit("/"), F.col("audio_document_id"))
    manifest_df = manifest_df.select(audio_file_path.alias("audio_filepath"),
                                     duration_udf(audio_file_path).alias("duration"),
                                     srt_to_text(manifest_df.content).alias("text"))
    manifest_df.write.mode("overwrite").format("json").save("nemo_manifest_jsonl")
    spark.read.format("json").load("nemo_manifest_jsonl").coalesce(1).write.mode("overwrite").format("json").save("nemo_manifest_jsonl_single")

def add_segment_split_to_text(text, segment_separator):

    # remove some symbols for better split into sentences
    text = (
        text.replace("\n", " ")
        .replace("\t", " ")
        .replace("…", "...")
        .replace("\\", " ")
        .replace("--", " -- ")
        .replace(". . .", "...")
    )

    # end of quoted speech - to be able to split sentences by full stop
    text = re.sub(r"([\.\?\!])([\"\'])", r"\g<2>\g<1> ", text)

    # remove extra space
    text = re.sub(r" +", " ", text)

    # remove normal brackets, square brackets and curly brackets
    text = re.sub(r'(\(.*?\))', ' ', text)
    text = re.sub(r'(\[.*?\])', ' ', text)
    text = re.sub(r'(\{.*?\})', ' ', text)
    
    # remove space in the middle of the lower case abbreviation to avoid splitting into separate sentences
    matches = re.findall(r'[a-z]\.\s[a-z]\.', text)
    for match in matches:
        text = text.replace(match, match.replace('. ', '.'))

    # find phrases in quotes
    with_quotes = re.finditer(r'“[A-Za-z ?]+.*?”', text)
    sentences = []
    last_idx = 0
    for m in with_quotes:
        match = m.group()
        match_idx = m.start()
        if last_idx < match_idx:
            sentences.append(text[last_idx:match_idx])
        sentences.append(match)
        last_idx = m.end()
    sentences.append(text[last_idx:])
    sentences = [s.strip() for s in sentences if s.strip()]

    # Read and split text by utterance (roughly, sentences)
    split_pattern = f"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|\!|\.”|\?”\!”)\s"

    new_sentences = []
    for sent in sentences:
        new_sentences.extend(regex.split(split_pattern, sent))
    sentences = [s.strip() for s in new_sentences if s.strip()]

    sentences = [" ".join(sent.split()) for sent in sentences]

    # remove any "- " at start of sentences
    sentences = [re.sub(r'^- ', "", sent) for sent in sentences]

    text = segment_separator.join(sentences)

    return text


@F.pandas_udf(T.StringType())
def srt_to_text(srt_file_contents: pd.Series) -> pd.Series:
    def helper(content: bytes) -> str:
        content = content.decode("utf-8") 
        try:
            total_transcript = " ".join(
                line.content.replace("\n", " ") for line in srt.parse(content)
            )
            return add_segment_split_to_text(total_transcript, "<segment_split>")
        except (srt.SRTParseError, srt.TimestampParseError) as exc:
            # Is this really the best way to log in a pandas UDF?
            print("WARNING: trouble parsing srt file content")
            print(exc)
            return ""

    return srt_file_contents.apply(helper)


@F.pandas_udf(T.DoubleType())
def duration_udf(audio_file_series: pd.Series) -> pd.Series:
    return audio_file_series.apply(lambda x: librosa.get_duration(path=x))

if __name__ == "__main__":
    fire.Fire(main)
