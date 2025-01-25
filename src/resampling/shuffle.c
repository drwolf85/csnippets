#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static inline void swap(uint8_t *a, uint8_t *b) {
	uint8_t tmp = *b;
	*b = *a;
	*a = tmp;
}

extern void shuffle(void *pt, const size_t len, const size_t sz) {
	size_t i, ii, j;
	uint8_t *v = (uint8_t *) pt;
	if (pt && v) {
		for (i = 0; i < len; i++) {
			ii = (size_t) random();
			ii %= len;
			for (j = 0; j < sz && i != ii; j++) {
				swap(&v[sz * i + j], &v[sz * ii + j]);
			}
		}
	}
}

#ifdef DEBUG
typedef struct dj_test {
	char name[32];
	char track[32];
} dj_t;

#include <time.h>

int main() {
	dj_t dj_list[] = {{"Armin van Bureen", "Bla bla bla"}, \
			  {"Clockartz", "Dawn"},\
			  {"ConceptArt", "Body Go Crack"}, \
			  {"Faizar", "Crab Nebula"},\
                          {"Headhunterz", "Dragonborn"}, \
			  {"Ottomix", "Plastica"}, \
			  {"Toneshifterz", "Shadows"}};
	srandom(time(NULL));
	for (int i = 0; i < 7; i++) printf("\t%s produced '%s'\n", dj_list[i].name, dj_list[i].track);
	printf("RANDOM SHUFFLING:\n");
	shuffle(dj_list, 7ULL, sizeof(dj_t));
	for (int i = 0; i < 7; i++) printf("\t%s produced '%s'\n", dj_list[i].name, dj_list[i].track);
	return 0;
}

#endif


