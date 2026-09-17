# A separate image; cachet:1.8 is never overwritten.
FROM cachet:1.8
WORKDIR /usr/src/app/cachet
COPY apply_candidate.py /tmp/apply_candidate.py
RUN python /tmp/apply_candidate.py
RUN make clean && make CFLAGS="-O3 -DNDEBUG -std=c++98" LFLAGS="-static"
