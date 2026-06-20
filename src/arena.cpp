#include <myx_grad/arena.h>
#include <cassert>

util::Arena::~Arena() {
    for (char* chunk : chunks_) {
        delete[] chunk;
    }
}
