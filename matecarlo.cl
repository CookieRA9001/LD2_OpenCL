float rand_uniform(uint* state) {
    (*state) *= ((*state) + 195439) * ((*state) + 124395) * ((*state) + 845921);
    return (float)(*state) / 4294967295.0f;
}

__kernel void matecarlo_fun(
    __global int* output,
    __global int* params
)
{
    int a = params[0], b = params[1], c = params[2];
    int xmin = params[3], xmax = params[4], ymin = params[5], ymax = params[6];
    int points = params[7];
    int i = get_global_id(0);
    uint state = (uint)(i + 1);

    int count = 0;

    for (int i = 0; i < points; i++)
    {
        float x = xmin + rand_uniform(&state) * (xmax - xmin);
        float y = ymin + rand_uniform(&state) * (ymax - ymin);

        float f = (x - a) * (x - b) * (x - c);

        if (f > 0.0f && y < f && y > 0.0f) count++;
        if (f < 0.0f && y > f && y < 0.0f) count--;
    }

    output[i] = count;
}