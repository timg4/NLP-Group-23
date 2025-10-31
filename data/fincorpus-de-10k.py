import datasets
from datasets.tasks import LanguageModeling

# TODO 
# - shows how to include metadata from a separate file: https://huggingface.co/datasets/SIA86/WaterFlowCountersRecognition/blob/e659c03dfc5e50dd08648b92d66b2f3f3ef560a4/WaterFlowCountersRecognition.py 
# - shows how to add and use custom kwargs that we could use for globbing filenames: https://discuss.huggingface.co/t/using-config-kwargs-within-the-load-dataset/32112/3

_DATA_URL = "https://huggingface.co/datasets/anhaltai/fincorpus-de-10k/resolve/main/data/corpus_safe_txt_only.zip"

ALL_COLLECTIONS_NAME = "all"

# Top-level directories' names
CONFIG_NAMES = {
    "Annual_reports",
    "BBK_monthly",
    "Base_prospectuses",
    "Final_terms",
    #  "IFRS",
    #  "Informational_materials",
    "Law",
    ALL_COLLECTIONS_NAME
}


# TODO
_DESCRIPTION = """\
We introduce a predominantly German corpus comprising 12.5k PDF documents (and 10.5k extracted txt files) sourced from the financial domain. The corresponding extracted textual data encompasses more than 165 million tokens derived predominantly from German, and to a lesser extent, bilingual documents. 
We provide detailed information about the document types included in the corpus, such as final terms, base prospectuses, annual reports, information materials, law documents, international financial reporting standards, and monthly reports from the Bundesbank, accompanied by comprehensive statistical analysis.
This version of the dataset excludes two collections, IFRS and Informational_materials, leaving only datasets definitely releasable with an open license.
"""

# TODO bibtex citation here
_CITATION = """  """


class FincorpusConfig(datasets.BuilderConfig):
    def __init__(self, generate_sentences=False, **kwargs):
        super(FincorpusConfig, self).__init__(
            version=datasets.Version("1.0.0"), **kwargs
        )


class Fincorpus(datasets.GeneratorBasedBuilder):
    # VERSION = datasets.Version('1.0.0')

    BUILDER_CONFIGS = [
        FincorpusConfig(name=config_name) for config_name in CONFIG_NAMES
    ]
    DEFAULT_CONFIG_NAME = ALL_COLLECTIONS_NAME

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "filename": datasets.Value("string"),
                    "text": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            #  citation=_CITATION,
            task_templates=[LanguageModeling(text_column="text")],
        )

    def _split_generators(self, dl_manager):
        #  config_urls = _DATA_URL[self.config.name]
        config_url = _DATA_URL
        arch_path = dl_manager.download(config_url)
        #  files_paths = dl_manager.download_and_extract(config_url)
        #  subdir = self.config.name
        #  clean_paths = [x for x in files_paths if x.startswith(subdir)]
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"files": dl_manager.iter_archive(arch_path)},
            ),
        ]

    def _path_belongs_to_collection(self, path: str):
        subfolder_name = self.config.name
        if subfolder_name == ALL_COLLECTIONS_NAME:
            return True

        if path.startswith("txt/" + subfolder_name):
            return True
        return False

    def _generate_examples(self, files):
        _id = 0
        for path, f in files:
            if not self._path_belongs_to_collection(path):
                continue
            text = f.read().decode("utf-8").strip()
            yield _id, {"text": text, "filename": path}
            _id += 1
