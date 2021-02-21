export CLASSPATH=/Users/zhengxiao/Documents/Codes/Library/stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar
cat test.jsonl.refs | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > test.hypo.target
cat test.jsonl.hyps | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > test.hypo.tokenized
files2rouge test.hypo.tokenized test.hypo.target