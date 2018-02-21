#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

import resq.config as config

DeclarativeBase = declarative_base()

session = None


def init():
    global session
    engine = create_engine('sqlite:///%s' % config.db_filename)
    Session = sessionmaker(expire_on_commit=False, autoflush=False)
    Session.configure(bind=engine)
    session = Session()

    DeclarativeBase.metadata.create_all(engine)
