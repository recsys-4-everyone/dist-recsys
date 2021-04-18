# -*- coding: utf-8 -*-


def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument("--type", type=str, help="job type")
    parser.add_argument("--index", type=int, help="job index")
