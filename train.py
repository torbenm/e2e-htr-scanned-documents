import sys
import argparse
from data.PaperNoteWords import PaperNoteWords
from lib.QuickExecutor import QuickExecutor
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument(
        '--gpu', help='Runs scripts on gpu. Default is cpu.', default=-1, type=int)
    parser.add_argument('--hardplacement', help='Disallow Softplacement, default is False',
                        action='store_true', default=False)
    parser.add_argument('--logplacement', help='Log Device placement',
                        action='store_true', default=False)
    parser.add_argument('--no-trans', help='Don\'t train transcriber',
                        action='store_true', default=False)
    parser.add_argument('--no-class', help='Don\'t train classifier',
                        action='store_true', default=False)
    parser.add_argument('--paper-notes', help='Add training on paper notes',
                        action='store_true', default=False)
    parser.add_argument('--timeline', default='')
    parser.add_argument('--legacy-transpose', help='Legacy: Perform transposing',
                        action='store_true', default=False)
    parser.add_argument(
        '--model-date', help='date to continue for', default='')
    parser.add_argument('--model-epoch', help='epoch to continue for',
                        default=0, type=int)
    args = parser.parse_args()

    qe = QuickExecutor(configName=args.config)
    qe.configure(softplacement=not args.hardplacement,
                 logplacement=args.logplacement, device=args.gpu)

    if args.paper_notes:
        paper_note_dataset = PaperNoteWords(
            meta=qe.dataset.meta, vocab=qe.dataset.vocab, data_config=qe.dataset.data_config, max_length=qe.dataset.max_length)
    if not args.no_trans:
        qe.add_train_transcriber()
        qe.add_transcription_validator()
        if args.paper_notes:
            qe.add_train_transcriber(dataset=paper_note_dataset, prefix='pn ')
            qe.add_transcription_validator(
                dataset=paper_note_dataset, prefix='pn ')
    if not args.no_class:
        qe.add_train_classifier()
        qe.add_class_validator()
        if args.paper_notes:
            qe.add_train_classifier(dataset=paper_note_dataset, prefix='pn ')
            qe.add_class_validator(
                dataset=paper_note_dataset, prefix='pn ')

    qe.add_saver()
    qe.add_summary()
    if args.model_date != "":
        qe.restore(args.model_date, args.model_epoch)
    qe()
