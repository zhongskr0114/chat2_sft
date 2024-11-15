# .utils

import jsonlines
# jsonl
def jsonl_load(file_path: str)->[]:
    ret = []
    with jsonlines.open(file_path) as lines:
        for line in lines:
            ret.append(line)
    return ret



