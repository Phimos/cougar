#ifdef T

#include "math.h"
#include "stdlib.h"

#define concat(a, b) a##_##b

#define monotonic_queue_pair(dtype) struct concat(monotonic_queue_pair, dtype)
#define monotonic_queue(dtype) struct concat(monotonic_queue, dtype)
#define method(name, dtype) concat(monotonic_queue_##name, dtype)

monotonic_queue_pair(T) {
    T value;
    size_t index;
};

monotonic_queue(T) {
    size_t size, capacity;
    size_t increasing;
    monotonic_queue_pair(T) * head, *tail;
    monotonic_queue_pair(T) * buffer, *buffer_end;
};

static inline void method(reset, T)(monotonic_queue(T) * queue) {
    queue->size = 0;
    queue->head = queue->tail = queue->buffer;
}

static inline monotonic_queue(T) * method(init, T)(size_t capacity, size_t increasing) {
    monotonic_queue(T)* queue = (monotonic_queue(T)*)malloc(sizeof(monotonic_queue(T)));
    queue->capacity = capacity;
    queue->increasing = increasing;
    queue->buffer = (monotonic_queue_pair(T)*)malloc(sizeof(monotonic_queue_pair(T)) * capacity);
    queue->buffer_end = queue->buffer + capacity - 1;
    method(reset, T)(queue);
    return queue;
}

static inline void method(free, T)(monotonic_queue(T) * queue) {
    free(queue->buffer);
    free(queue);
}

static inline monotonic_queue_pair(T) * method(next, T)(monotonic_queue(T) * queue, monotonic_queue_pair(T) * pointer) {
    return (pointer == queue->buffer_end) ? (queue->buffer) : (pointer + 1);
}

static inline monotonic_queue_pair(T) * method(prev, T)(monotonic_queue(T) * queue, monotonic_queue_pair(T) * pointer) {
    return (pointer == queue->buffer) ? (queue->buffer_end) : (pointer - 1);
}

static inline T method(front_value, T)(monotonic_queue(T) * queue) {
    return queue->head->value;
}

static inline size_t method(front_index, T)(monotonic_queue(T) * queue) {
    return queue->head->index;
}

static inline T method(back_value, T)(monotonic_queue(T) * queue) {
    return queue->tail->value;
}

static inline size_t method(back_index, T)(monotonic_queue(T) * queue) {
    return queue->tail->index;
}

static inline void method(pop_front, T)(monotonic_queue(T) * queue) {
    queue->head = method(next, T)(queue, queue->head);
    queue->size--;
}

static inline void method(pop_back, T)(monotonic_queue(T) * queue) {
    queue->tail = method(prev, T)(queue, queue->tail);
    queue->size--;
}

static inline void method(push_back, T)(monotonic_queue(T) * queue, T value, size_t index) {
    if (queue->size == 0) {
        queue->head = queue->tail = queue->buffer;
    } else {
        queue->tail = method(next, T)(queue, queue->tail);
    }
    queue->tail->value = value;
    queue->tail->index = index;
    queue->size++;
}

static inline void method(push, T)(monotonic_queue(T) * queue, T value, size_t index) {
    if (queue->increasing) {
        if (queue->size > 0 && method(front_value, T)(queue) >= value) {
            method(reset, T)(queue);
        } else {
            while (queue->size > 0 && method(back_value, T)(queue) >= value) {
                method(pop_back, T)(queue);
            }
        }
    } else {
        if (queue->size > 0 && method(front_value, T)(queue) <= value) {
            method(reset, T)(queue);
        } else {
            while (queue->size > 0 && method(back_value, T)(queue) <= value) {
                method(pop_back, T)(queue);
            }
        }
    }
    method(push_back, T)(queue, value, index);
}

#undef method
#undef monotonic_queue_pair
#undef monotonic_queue
#undef concat

#endif