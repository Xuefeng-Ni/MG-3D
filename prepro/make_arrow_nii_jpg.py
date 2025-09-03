import os
from collections import Counter, defaultdict

import pandas as pd
import pyarrow as pa
from tqdm import tqdm

from glossary import normalize_word


def statistics(iid2captions, iid2split):
    all_images = {"train": [], "val": [], "test": []}
    all_texts = {"train": [], "val": [], "test": []}

    for iid, texts in iid2captions.items():
        split = iid2split[iid]
        all_images[split].append(iid)
        all_texts[split].extend(texts)

    for split, images in all_images.items():
        print(f"+ {split} set: {len(images)} images")

    for split, texts in all_texts.items():
        lengths = [len(text.split()) for text in texts]
        avg_len = sum(lengths) / len(lengths)
        print(f"+ {split} set: {len(texts)} texts")
        print(f"+ {split} set: {avg_len} words in average.")
        lengths = [length // 10 * 10 for length in lengths]
        print(Counter(lengths))

def path2rest(path, iid2captions, iid2split):
    name = path

    binary_list = []
    img_path = os.listdir(name)
    for i in range(len(img_path)):
        if img_path[i][-4:] == '.jpg':
            path = name + '/' + img_path[i]
            with open(path, "rb") as fp:
                binary = fp.read()
            binary_list.append(binary)
    
    captions = iid2captions[name]
    split = iid2split[name]
    return [binary_list, captions, name, split]

def make_arrow(data, dataset_name, save_dir):
    print(f"+ Pre-processing {dataset_name}...")
    iid2captions = defaultdict(list)
    iid2split = dict()

    for split, split_data in data.items():
        for sample in split_data:
            iid2captions[sample["img_path"]].extend(sample["texts"])
            iid2split[sample["img_path"]] = split

    path = len(iid2captions)
    caption_paths = [path for path in iid2captions if os.path.exists(path)]
    print(f"+ {len(caption_paths)} images / {path} annotations")
    statistics(iid2captions, iid2split)
    import pdb
    pdb.set_trace()
    bs = [path2rest(path, iid2captions, iid2split) for path in tqdm(caption_paths)]

    for split in ["train", "val", "test"]:
        batches = [b for b in bs if b[-1] == split]
        dataframe = pd.DataFrame(batches, columns=["image", "caption", "image_id", "split"])
        table = pa.Table.from_pandas(dataframe)
        os.makedirs(save_dir, exist_ok=True)
        with pa.OSFile(f"{save_dir}/{dataset_name}_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)


def path2rest_mimic_cxr(path, iid2captions, iid2chexpert, iid2split):
    name = path
    with open(path, "rb") as fp:
        binary = fp.read()
    captions = iid2captions[name]
    chexpert = iid2chexpert[name]
    split = iid2split[name]
    return [binary, captions, name, chexpert, split]


def make_arrow_mimic_cxr(data, dataset_name, save_dir):
    print(f"+ Pre-processing {dataset_name}...")
    iid2captions = defaultdict(list)
    iid2chexpert = defaultdict(list)
    iid2split = dict()

    for split, split_data in data.items():
        for sample in split_data:
            iid2captions[sample["img_path"]].extend(sample["texts"])
            iid2chexpert[sample["img_path"]].extend(sample["chexpert"])
            iid2split[sample["img_path"]] = split

    path = len(iid2captions)
    caption_paths = [path for path in iid2captions if os.path.exists(path)]
    print(f"+ {len(caption_paths)} images / {path} annotations")
    statistics(iid2captions, iid2split)
    import pdb
    pdb.set_trace()
    bs = [path2rest_mimic_cxr(path, iid2captions, iid2chexpert, iid2split) for path in tqdm(caption_paths)]

    for split in ["train", "val", "test"]:
        batches = [b for b in bs if b[-1] == split]
        dataframe = pd.DataFrame(batches, columns=["image", "caption", "image_id", "chexpert", "split"])
        table = pa.Table.from_pandas(dataframe)
        os.makedirs(save_dir, exist_ok=True)
        with pa.OSFile(f"{save_dir}/{dataset_name}_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)


def get_score(occurences):
    return 1.0


def path2rest_vqa(path, split, annotations, label2ans):
    with open(path, "rb") as fp:
        binary = fp.read()

    iid = path
    _annotation = annotations[split][iid]
    _annotation = list(_annotation.items())
    qids, qas = [a[0] for a in _annotation], [a[1] for a in _annotation]
    questions = [qa[0] for qa in qas]
    answers = [qa[1] for qa in qas]
    answer_labels = [a["labels"] for a in answers]
    answer_scores = [a["scores"] for a in answers]
    question_types = [a["answer_type"] for a in answers]
    answers = [[label2ans[l] for l in al] for al in answer_labels]

    return [binary, questions, answers, answer_labels, answer_scores, iid, qids, question_types, split]


def make_arrow_vqa(data, dataset_name, save_dir):
    questions_train, questions_val, questions_test = data["train"], data["val"], data["test"]

    # Record Questions
    annotations = dict()
    for split, questions in zip(["train", "val", "test"], [questions_train, questions_val, questions_test]):
        _annotation = defaultdict(dict)
        for q in tqdm(questions):
            _annotation[q["img_path"]][q["qid"]] = [q["question"]]
        annotations[split] = _annotation

    # Construct Vocabulary
    all_major_answers = list()
    for split, questions in zip(["train", "val", "test"], [questions_train, questions_val, questions_test]):
        for q in tqdm(questions):
            all_major_answers.append(str(q["answer"]).lower())
    all_major_answers = [normalize_word(word) for word in tqdm(all_major_answers)]
    counter = {k: v for k, v in Counter(all_major_answers).items() if v >= 0}
    ans2label = {k: i for i, k in enumerate(counter.keys())}
    label2ans = list(counter.keys())
    print("Label size ({}): {}.".format(dataset_name, len(ans2label)))

    # Record Answers
    for split, questions in zip(["train", "val", "test"], [questions_train, questions_val, questions_test]):
        _annotation = annotations[split]
        for q in tqdm(questions):
            answers = normalize_word(str(q["answer"]).lower())
            answer_count = {}
            answer_count[answers] = answer_count.get(answers, 0) + 1
            labels = []
            scores = []
            for answer in answer_count:
                assert answer in ans2label
                labels.append(ans2label[answer])
                score = get_score(answer_count[answer])
                scores.append(score)
            assert q['answer_type'].strip().lower() == "closed" or q['answer_type'].strip().lower() == "open"
            answer_type = 0 if q['answer_type'].strip().lower() == "closed" else 1
            _annotation[q["img_path"]][q["qid"]].append(
                {"labels": labels, "scores": scores, "answer_type": answer_type})

    # Write to the files
    for split in ["train", "val", "test"]:
        annot = annotations[split]
        annot_paths = [path for path in annot if os.path.exists(path)]
        assert len(annot_paths) == len(annot) or len(annot_paths) == len(annot) - 1
        print("{} set: {} images, {} questions".format(split,
                                                       len(annot),
                                                       len([vv for k, v in annot.items() for kk, vv in v.items()])))

        bs = [
            path2rest_vqa(path, split, annotations, label2ans) for path in tqdm(annot_paths)
        ]
        dataframe = pd.DataFrame(
            bs,
            columns=[
                "image",
                "questions",
                "answers",
                "answer_labels",
                "answer_scores",
                "image_id",
                "question_id",
                "answer_type",
                "split",
            ],
        )
        table = pa.Table.from_pandas(dataframe)

        os.makedirs(save_dir, exist_ok=True)
        with pa.OSFile(f"{save_dir}/{dataset_name}_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)


def path2rest_melinda(path, iid2captions, iid2i_meth, iid2p_meth, iid2i_meth_label, iid2p_meth_label, iid2split):
    name = path
    with open(path, "rb") as fp:
        binary = fp.read()
    captions = iid2captions[name]
    i_meth = iid2i_meth[name]
    p_meth = iid2p_meth[name]
    i_meth_label = iid2i_meth_label[name]
    p_meth_label = iid2p_meth_label[name]
    assert len(captions) == len(i_meth)
    assert len(captions) == len(p_meth)
    assert len(captions) == len(i_meth_label)
    assert len(captions) == len(p_meth_label)
    split = iid2split[name]
    return [binary, captions, name, i_meth, p_meth, i_meth_label, p_meth_label, split]


def make_arrow_melinda(data, dataset_name, save_dir):
    print(f"+ Pre-processing {dataset_name}...")
    iid2captions = defaultdict(list)
    iid2i_meth = defaultdict(list)
    iid2p_meth = defaultdict(list)
    iid2i_meth_label = defaultdict(list)
    iid2p_meth_label = defaultdict(list)
    iid2split = dict()

    for split, split_data in data.items():
        for sample in split_data:
            iid2captions[sample["img_path"]].extend(sample["texts"])
            iid2split[sample["img_path"]] = split
            iid2i_meth[sample["img_path"]].append(sample["i_meth"])
            iid2p_meth[sample["img_path"]].append(sample["p_meth"])
            iid2i_meth_label[sample["img_path"]].append(sample["i_meth_label"])
            iid2p_meth_label[sample["img_path"]].append(sample["p_meth_label"])

    i_meth_set = set([vv for k, v in iid2i_meth.items() for vv in v])
    i_meth_label_set = set([vv for k, v in iid2i_meth_label.items() for vv in v])
    p_meth_set = set([vv for k, v in iid2p_meth.items() for vv in v])
    p_meth_label_set = set([vv for k, v in iid2p_meth_label.items() for vv in v])

    i_meth_set = sorted(i_meth_set)
    i_meth_label_set = sorted(i_meth_label_set)
    p_meth_set = sorted(p_meth_set)
    p_meth_label_set = sorted(p_meth_label_set)

    i_meth_dict = {j: i for i, j in enumerate(i_meth_set)}
    p_meth_dict = {j: i for i, j in enumerate(p_meth_set)}
    i_meth_label_dict = {j: i for i, j in enumerate(i_meth_label_set)}
    p_meth_label_dict = {j: i for i, j in enumerate(p_meth_label_set)}

    iid2i_meth = {k: [i_meth_dict[vv] for vv in v] for k, v in iid2i_meth.items()}
    iid2p_meth = {k: [p_meth_dict[vv] for vv in v] for k, v in iid2p_meth.items()}
    iid2i_meth_label = {k: [i_meth_label_dict[vv] for vv in v] for k, v in iid2i_meth_label.items()}
    iid2p_meth_label = {k: [p_meth_label_dict[vv] for vv in v] for k, v in iid2p_meth_label.items()}

    path = len(iid2captions)
    caption_paths = [path for path in iid2captions if os.path.exists(path)]
    print(f"+ {len(caption_paths)} images / {path} annotations")
    statistics(iid2captions, iid2split)
    bs = [path2rest_melinda(path, iid2captions, iid2i_meth, iid2p_meth, iid2i_meth_label, iid2p_meth_label, iid2split)
          for path in tqdm(caption_paths)]

    for split in ["train", "val", "test"]:
        batches = [b for b in bs if b[-1] == split]
        dataframe = pd.DataFrame(batches, columns=["image", "caption", "image_id", "i_meth", "p_meth", "i_meth_label",
                                                   "p_meth_label", "split"])
        table = pa.Table.from_pandas(dataframe)
        os.makedirs(save_dir, exist_ok=True)
        with pa.OSFile(f"{save_dir}/{dataset_name}_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)


def path2rest_chexpert(path, iid2captions, iid2chexpert, iid2split):
    name = path
    with open(path, "rb") as fp:
        binary = fp.read()
    captions = iid2captions[name]
    chexpert = iid2chexpert[name]
    split = iid2split[name]
    return [binary, captions, name, chexpert, split]


def make_arrow_chexpert(data, dataset_name, save_dir):
    print(f"+ Pre-processing {dataset_name}...")
    for split, split_data in data.items():
        iid2captions = defaultdict(list)
        iid2chexpert = defaultdict(list)
        iid2split = dict()

        for sample in split_data:
            iid2captions[sample["img_path"]].extend(sample["texts"])
            iid2chexpert[sample["img_path"]].extend(sample["chexpert"])
            iid2split[sample["img_path"]] = split

        path = len(iid2captions)
        caption_paths = [path for path in iid2captions if os.path.exists(path)]
        print(f"+ {len(caption_paths)} images / {path} annotations")
        bs = [path2rest_chexpert(path, iid2captions, iid2chexpert, iid2split) for path in tqdm(caption_paths)]

        batches = [b for b in bs if b[-1] == split]
        dataframe = pd.DataFrame(batches, columns=["image", "caption", "image_id", "chexpert", "split"])
        table = pa.Table.from_pandas(dataframe)
        os.makedirs(save_dir, exist_ok=True)
        with pa.OSFile(f"{save_dir}/{dataset_name}_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)


def path2rest_pnsa_pneumonia(path, iid2captions, iid2pnsa_pneumonia, iid2split):
    name = path

    with open(path, "rb") as fp:
        binary = fp.read()
    captions = iid2captions[name]
    pnsa_pneumonia = iid2pnsa_pneumonia[name]
    split = iid2split[name]
    return [binary, captions, name, pnsa_pneumonia, split]


def make_arrow_pnsa_pneumonia(data, dataset_name, save_dir):
    print(f"+ Pre-processing {dataset_name}...")
    for split, split_data in data.items():
        iid2captions = defaultdict(list)
        iid2pnsa_pneumonia = defaultdict(list)
        iid2split = dict()

        for sample in split_data:
            iid2captions[sample["img_path"]].extend(sample["texts"])
            iid2pnsa_pneumonia[sample["img_path"]].extend(sample["pnsa_pneumonia"])
            iid2split[sample["img_path"]] = split

        path = len(iid2captions)
        caption_paths = [path for path in iid2captions if os.path.exists(path)]
        print(f"+ {len(caption_paths)} images / {path} annotations")
        bs = [path2rest_pnsa_pneumonia(path, iid2captions, iid2pnsa_pneumonia, iid2split) for path in
              tqdm(caption_paths)]

        batches = [b for b in bs if b[-1] == split]
        dataframe = pd.DataFrame(batches, columns=["image", "caption", "image_id", "pnsa_pneumonia", "split"])
        table = pa.Table.from_pandas(dataframe)
        os.makedirs(save_dir, exist_ok=True)
        with pa.OSFile(f"{save_dir}/{dataset_name}_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)


def path2rest_clm_mimic_cxr(path, iid2captions, iid2findings, iid2impression, iid2chexpert, iid2split):
    name = path
    with open(path, "rb") as fp:
        binary = fp.read()
    captions = iid2captions[name]
    findings = iid2findings[name]
    impression = iid2impression[name]
    assert len(captions) == 1 and len(impression) == 1
    chexpert = iid2chexpert[name]
    split = iid2split[name]
    return [binary, captions, name, findings, impression, chexpert, split]

def path2rest_clm_ctrg_chest_nitfy(path, iid2captions, iid2findings, iid2impression, iid2findings_sen, iid2impression_sen, iid2split):
    name = path

    captions = iid2captions[name]
    findings = iid2findings[name]
    impression = iid2impression[name]
    findings_sen = iid2findings_sen[name]
    impression_sen = iid2impression_sen[name]
    assert len(captions) == 1 and len(impression) == 1
    split = iid2split[name]
    return [captions, name, findings, impression, findings_sen, impression_sen, split]

def path2rest_clm_ctrg_chest(path, iid2captions, iid2findings, iid2impression, iid2findings_sen, iid2impression_sen, iid2split):
    name = path

    binary_list = []
    img_path = os.listdir(name)
    for i in range(len(img_path)):
        if img_path[i][-4:] == '.jpg':
            path = name + '/' + img_path[i]
            with open(path, "rb") as fp:
                binary = fp.read()
            binary_list.append(binary)

    captions = iid2captions[name]
    findings = iid2findings[name]
    impression = iid2impression[name]
    findings_sen = iid2findings_sen[name]
    impression_sen = iid2impression_sen[name]
    assert len(captions) == 1 and len(impression) == 1
    split = iid2split[name]
    return [binary_list, captions, name, findings, impression, findings_sen, impression_sen, split]


def make_arrow_clm_mimic_cxr(data, dataset_name, save_dir):
    print(f"+ Pre-processing {dataset_name}...")
    iid2captions = defaultdict(list)
    iid2findings = defaultdict(list)
    iid2impression = defaultdict(list)
    iid2chexpert = defaultdict(list)
    iid2split = dict()

    for split, split_data in data.items():
        for sample in split_data:
            iid2captions[sample["img_path"]].extend(sample["texts"])
            iid2findings[sample["img_path"]].extend(sample["findings"])
            iid2impression[sample["img_path"]].extend(sample["impression"])
            iid2chexpert[sample["img_path"]].extend(sample["chexpert"])
            iid2split[sample["img_path"]] = split

    path = len(iid2captions)
    caption_paths = [path for path in iid2captions if os.path.exists(path)]
    print(f"+ {len(caption_paths)} images / {path} annotations")
    statistics(iid2captions, iid2split)
    bs = [path2rest_clm_mimic_cxr(path, iid2captions, iid2findings, iid2impression, iid2chexpert, iid2split) for path in
          tqdm(caption_paths)]

    for split in ["train", "val", "test"]:
        batches = [b for b in bs if b[-1] == split]
        dataframe = pd.DataFrame(batches, columns=["image", "caption", "image_id", "findings", "impression",
                                                   "chexpert", "split"])
        table = pa.Table.from_pandas(dataframe)
        os.makedirs(save_dir, exist_ok=True)
        with pa.OSFile(f"{save_dir}/{dataset_name}_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)


def make_arrow_clm_ctrg_chest(data, dataset_name, save_dir):
    print(f"+ Pre-processing {dataset_name}...")
    iid2captions = defaultdict(list)
    iid2findings = defaultdict(list)
    iid2impression = defaultdict(list)
    iid2findings_sen = defaultdict(list)
    iid2impression_sen = defaultdict(list)
    iid2split = dict()

    for split, split_data in data.items():
        for sample in split_data:
            iid2captions[os.path.split(sample["img_path"][0])[0]].extend(sample["texts"])
            iid2findings[os.path.split(sample["img_path"][0])[0]].extend(sample["findings"])
            iid2impression[os.path.split(sample["img_path"][0])[0]].extend(sample["impression"])
            iid2findings_sen[os.path.split(sample["img_path"][0])[0]].extend(sample["findings_sen"])
            iid2impression_sen[os.path.split(sample["img_path"][0])[0]].extend(sample["impression_sen"])
            iid2split[os.path.split(sample["img_path"][0])[0]] = split

    path = len(iid2captions)
    caption_paths = [path for path in iid2captions if os.path.exists(path)]
    print(f"+ {len(caption_paths)} images / {path} annotations")
    statistics(iid2captions, iid2split)
    bs = [path2rest_clm_ctrg_chest(path, iid2captions, iid2findings, iid2impression, iid2findings_sen, iid2impression_sen, iid2split) for path in
          tqdm(caption_paths)]

    for split in ["train", "val", "test"]:
        batches = [b for b in bs if b[-1] == split]
        dataframe = pd.DataFrame(batches, columns=["image", "caption", "image_id", "findings", "impression", "findings_sen", "impression_sen",
                                                   "split"])
        table = pa.Table.from_pandas(dataframe)
        os.makedirs(save_dir, exist_ok=True)
        with pa.OSFile(f"{save_dir}/{dataset_name}_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)


def make_arrow_clm_ctrg_chest_nitfy(data, dataset_name, save_dir):
    print(f"+ Pre-processing {dataset_name}...")
    iid2captions = defaultdict(list)
    iid2findings = defaultdict(list)
    iid2impression = defaultdict(list)
    iid2findings_sen = defaultdict(list)
    iid2impression_sen = defaultdict(list)
    iid2split = dict()

    for split, split_data in data.items():
        for sample in split_data:
            iid2captions[sample["img_path"]].extend(sample["texts"])
            iid2findings[sample["img_path"]].extend(sample["findings"])
            iid2impression[sample["img_path"]].extend(sample["impression"])
            iid2findings_sen[sample["img_path"]].extend(sample["findings_sen"])
            iid2impression_sen[sample["img_path"]].extend(sample["impression_sen"])
            iid2split[sample["img_path"]] = split
            
    path = len(iid2captions)
    caption_paths = [path for path in iid2captions if os.path.exists(path)]
    print(f"+ {len(caption_paths)} images / {path} annotations")
    statistics(iid2captions, iid2split)
    bs = [path2rest_clm_ctrg_chest_nitfy(path, iid2captions, iid2findings, iid2impression, iid2findings_sen, iid2impression_sen, iid2split) for path in
          tqdm(caption_paths)]

    for split in ["train", "val", "test"]:
        batches = [b for b in bs if b[-1] == split]
        dataframe = pd.DataFrame(batches, columns=["caption", "image_id", "findings", "impression", "findings_sen", "impression_sen",
                                                   "split"])

        table = pa.Table.from_pandas(dataframe)
        os.makedirs(save_dir, exist_ok=True)
        with pa.OSFile(f"{save_dir}/{dataset_name}_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)


def path2rest_text_classification(guid, iid2text_a, iid2labels, iid2split):
    text_a = iid2text_a[guid]
    labels = iid2labels[guid]
    split = iid2split[guid]
    assert len(text_a) == 1
    return [text_a, guid, labels, split]


def make_arrow_text_classification(data, dataset_name, save_dir):
    print(f"+ Pre-processing {dataset_name}...")
    iid2text_a = defaultdict(list)
    iid2labels = dict()
    iid2split = dict()

    for split, split_data in data.items():
        for sample in split_data:
            iid2text_a[sample["guid"]].extend(sample["text_a"])
            iid2labels[sample["guid"]] = sample["label"]
            iid2split[sample["guid"]] = split

    statistics(iid2text_a, iid2split)
    bs = [path2rest_text_classification(guid, iid2text_a, iid2labels, iid2split) for guid in tqdm(iid2text_a)]

    for split in ["train", "val", "test"]:
        batches = [b for b in bs if b[-1] == split]
        dataframe = pd.DataFrame(batches, columns=["text_a", "guid", "label", "split"])
        table = pa.Table.from_pandas(dataframe)
        os.makedirs(save_dir, exist_ok=True)
        with pa.OSFile(f"{save_dir}/{dataset_name}_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)


def path2rest_nli(guid, iid2text_a, iid2text_b, iid2text, iid2labels, iid2split):
    text_a = iid2text_a[guid]
    text_b = iid2text_b[guid]
    text = iid2text[guid]
    labels = iid2labels[guid]
    split = iid2split[guid]
    assert len(text_a) == 1
    assert len(text_b) == 1
    assert len(text) == 1
    return [text_a, text_b, text, guid, labels, split]


def make_arrow_text_nli(data, dataset_name, save_dir):
    print(f"+ Pre-processing {dataset_name}...")
    iid2text_a = defaultdict(list)
    iid2text_b = defaultdict(list)
    iid2text = defaultdict(list)
    iid2labels = dict()
    iid2split = dict()

    for split, split_data in data.items():
        for sample in split_data:
            iid2text_a[sample["guid"]].extend(sample["text_a"])
            iid2text_b[sample["guid"]].extend(sample["text_b"])
            iid2text[sample["guid"]].extend(sample["text"])
            iid2labels[sample["guid"]] = sample["label"]
            iid2split[sample["guid"]] = split

    statistics(iid2text_a, iid2split)
    statistics(iid2text_b, iid2split)
    statistics(iid2text, iid2split)
    bs = [path2rest_nli(guid, iid2text_a, iid2text_b, iid2text, iid2labels, iid2split) for guid in tqdm(iid2text_a)]

    for split in ["train", "val", "test"]:
        batches = [b for b in bs if b[-1] == split]
        dataframe = pd.DataFrame(batches, columns=["text_a", "text_b", "text", "guid", "label", "split"])
        table = pa.Table.from_pandas(dataframe)
        os.makedirs(save_dir, exist_ok=True)
        with pa.OSFile(f"{save_dir}/{dataset_name}_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
