module test;

public int add_test(int a, int b) { return b + a; }

[numthreads(16, 16, 1)]
[shader("compute")]
void main(uint3 global_id : SV_DispatchThreadID, uint3 local_id : SV_GroupThreadID) { }