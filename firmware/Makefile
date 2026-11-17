# Host build of the portable forward filter. The CMSIS-NN backend is selected on target
# with CFLAGS += -DHMM_USE_CMSIS_NN and the CMSIS-NN include path; it is not built here.
CC ?= cc
CFLAGS ?= -O2
CFLAGS += -std=c99 -Wall -Wextra -pedantic -Iinclude
ifeq ($(WERROR),1)
CFLAGS += -Werror
endif
BUILD := build
SRC := src/hmm_filter.c src/hmm_fc_s8_ref.c src/hmm_exp_lut.c

.PHONY: all test lib clean

all: $(BUILD)/test_hmm_filter

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/test_hmm_filter: $(SRC) tests/test_hmm_filter.c include/*.h tests/hmm_vectors_generated.h | $(BUILD)
	$(CC) $(CFLAGS) -Itests -o $@ $(SRC) tests/test_hmm_filter.c

lib: $(BUILD)/libhmmfilter.so

$(BUILD)/libhmmfilter.so: $(SRC) include/*.h | $(BUILD)
	$(CC) $(CFLAGS) -fPIC -shared -o $@ $(SRC)

test: $(BUILD)/test_hmm_filter
	./$(BUILD)/test_hmm_filter

clean:
	rm -rf $(BUILD)
