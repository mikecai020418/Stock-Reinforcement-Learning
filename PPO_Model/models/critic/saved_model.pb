��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.15.02v2.15.0-rc1-8-g6887368d6d48��
�
"Adam/v/critic_network/dense_5/biasVarHandleOp*
_output_shapes
: *3

debug_name%#Adam/v/critic_network/dense_5/bias/*
dtype0*
shape:*3
shared_name$"Adam/v/critic_network/dense_5/bias
�
6Adam/v/critic_network/dense_5/bias/Read/ReadVariableOpReadVariableOp"Adam/v/critic_network/dense_5/bias*
_output_shapes
:*
dtype0
�
"Adam/m/critic_network/dense_5/biasVarHandleOp*
_output_shapes
: *3

debug_name%#Adam/m/critic_network/dense_5/bias/*
dtype0*
shape:*3
shared_name$"Adam/m/critic_network/dense_5/bias
�
6Adam/m/critic_network/dense_5/bias/Read/ReadVariableOpReadVariableOp"Adam/m/critic_network/dense_5/bias*
_output_shapes
:*
dtype0
�
$Adam/v/critic_network/dense_5/kernelVarHandleOp*
_output_shapes
: *5

debug_name'%Adam/v/critic_network/dense_5/kernel/*
dtype0*
shape:	�*5
shared_name&$Adam/v/critic_network/dense_5/kernel
�
8Adam/v/critic_network/dense_5/kernel/Read/ReadVariableOpReadVariableOp$Adam/v/critic_network/dense_5/kernel*
_output_shapes
:	�*
dtype0
�
$Adam/m/critic_network/dense_5/kernelVarHandleOp*
_output_shapes
: *5

debug_name'%Adam/m/critic_network/dense_5/kernel/*
dtype0*
shape:	�*5
shared_name&$Adam/m/critic_network/dense_5/kernel
�
8Adam/m/critic_network/dense_5/kernel/Read/ReadVariableOpReadVariableOp$Adam/m/critic_network/dense_5/kernel*
_output_shapes
:	�*
dtype0
�
"Adam/v/critic_network/dense_4/biasVarHandleOp*
_output_shapes
: *3

debug_name%#Adam/v/critic_network/dense_4/bias/*
dtype0*
shape:�*3
shared_name$"Adam/v/critic_network/dense_4/bias
�
6Adam/v/critic_network/dense_4/bias/Read/ReadVariableOpReadVariableOp"Adam/v/critic_network/dense_4/bias*
_output_shapes	
:�*
dtype0
�
"Adam/m/critic_network/dense_4/biasVarHandleOp*
_output_shapes
: *3

debug_name%#Adam/m/critic_network/dense_4/bias/*
dtype0*
shape:�*3
shared_name$"Adam/m/critic_network/dense_4/bias
�
6Adam/m/critic_network/dense_4/bias/Read/ReadVariableOpReadVariableOp"Adam/m/critic_network/dense_4/bias*
_output_shapes	
:�*
dtype0
�
$Adam/v/critic_network/dense_4/kernelVarHandleOp*
_output_shapes
: *5

debug_name'%Adam/v/critic_network/dense_4/kernel/*
dtype0*
shape:
��*5
shared_name&$Adam/v/critic_network/dense_4/kernel
�
8Adam/v/critic_network/dense_4/kernel/Read/ReadVariableOpReadVariableOp$Adam/v/critic_network/dense_4/kernel* 
_output_shapes
:
��*
dtype0
�
$Adam/m/critic_network/dense_4/kernelVarHandleOp*
_output_shapes
: *5

debug_name'%Adam/m/critic_network/dense_4/kernel/*
dtype0*
shape:
��*5
shared_name&$Adam/m/critic_network/dense_4/kernel
�
8Adam/m/critic_network/dense_4/kernel/Read/ReadVariableOpReadVariableOp$Adam/m/critic_network/dense_4/kernel* 
_output_shapes
:
��*
dtype0
�
"Adam/v/critic_network/dense_3/biasVarHandleOp*
_output_shapes
: *3

debug_name%#Adam/v/critic_network/dense_3/bias/*
dtype0*
shape:�*3
shared_name$"Adam/v/critic_network/dense_3/bias
�
6Adam/v/critic_network/dense_3/bias/Read/ReadVariableOpReadVariableOp"Adam/v/critic_network/dense_3/bias*
_output_shapes	
:�*
dtype0
�
"Adam/m/critic_network/dense_3/biasVarHandleOp*
_output_shapes
: *3

debug_name%#Adam/m/critic_network/dense_3/bias/*
dtype0*
shape:�*3
shared_name$"Adam/m/critic_network/dense_3/bias
�
6Adam/m/critic_network/dense_3/bias/Read/ReadVariableOpReadVariableOp"Adam/m/critic_network/dense_3/bias*
_output_shapes	
:�*
dtype0
�
$Adam/v/critic_network/dense_3/kernelVarHandleOp*
_output_shapes
: *5

debug_name'%Adam/v/critic_network/dense_3/kernel/*
dtype0*
shape:	�*5
shared_name&$Adam/v/critic_network/dense_3/kernel
�
8Adam/v/critic_network/dense_3/kernel/Read/ReadVariableOpReadVariableOp$Adam/v/critic_network/dense_3/kernel*
_output_shapes
:	�*
dtype0
�
$Adam/m/critic_network/dense_3/kernelVarHandleOp*
_output_shapes
: *5

debug_name'%Adam/m/critic_network/dense_3/kernel/*
dtype0*
shape:	�*5
shared_name&$Adam/m/critic_network/dense_3/kernel
�
8Adam/m/critic_network/dense_3/kernel/Read/ReadVariableOpReadVariableOp$Adam/m/critic_network/dense_3/kernel*
_output_shapes
:	�*
dtype0
�
learning_rateVarHandleOp*
_output_shapes
: *

debug_namelearning_rate/*
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
�
	iterationVarHandleOp*
_output_shapes
: *

debug_name
iteration/*
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
�
critic_network/dense_5/biasVarHandleOp*
_output_shapes
: *,

debug_namecritic_network/dense_5/bias/*
dtype0*
shape:*,
shared_namecritic_network/dense_5/bias
�
/critic_network/dense_5/bias/Read/ReadVariableOpReadVariableOpcritic_network/dense_5/bias*
_output_shapes
:*
dtype0
�
critic_network/dense_5/kernelVarHandleOp*
_output_shapes
: *.

debug_name critic_network/dense_5/kernel/*
dtype0*
shape:	�*.
shared_namecritic_network/dense_5/kernel
�
1critic_network/dense_5/kernel/Read/ReadVariableOpReadVariableOpcritic_network/dense_5/kernel*
_output_shapes
:	�*
dtype0
�
critic_network/dense_4/biasVarHandleOp*
_output_shapes
: *,

debug_namecritic_network/dense_4/bias/*
dtype0*
shape:�*,
shared_namecritic_network/dense_4/bias
�
/critic_network/dense_4/bias/Read/ReadVariableOpReadVariableOpcritic_network/dense_4/bias*
_output_shapes	
:�*
dtype0
�
critic_network/dense_4/kernelVarHandleOp*
_output_shapes
: *.

debug_name critic_network/dense_4/kernel/*
dtype0*
shape:
��*.
shared_namecritic_network/dense_4/kernel
�
1critic_network/dense_4/kernel/Read/ReadVariableOpReadVariableOpcritic_network/dense_4/kernel* 
_output_shapes
:
��*
dtype0
�
critic_network/dense_3/biasVarHandleOp*
_output_shapes
: *,

debug_namecritic_network/dense_3/bias/*
dtype0*
shape:�*,
shared_namecritic_network/dense_3/bias
�
/critic_network/dense_3/bias/Read/ReadVariableOpReadVariableOpcritic_network/dense_3/bias*
_output_shapes	
:�*
dtype0
�
critic_network/dense_3/kernelVarHandleOp*
_output_shapes
: *.

debug_name critic_network/dense_3/kernel/*
dtype0*
shape:	�*.
shared_namecritic_network/dense_3/kernel
�
1critic_network/dense_3/kernel/Read/ReadVariableOpReadVariableOpcritic_network/dense_3/kernel*
_output_shapes
:	�*
dtype0
�
serving_default_input_1Placeholder*/
_output_shapes
:���������*
dtype0*$
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1critic_network/dense_3/kernelcritic_network/dense_3/biascritic_network/dense_4/kernelcritic_network/dense_4/biascritic_network/dense_5/kernelcritic_network/dense_5/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_140858

NoOpNoOp
�*
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�*
value�*B�* B�)
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
fc1
	fc2

flatten
q
	optimizer
loss

signatures*
.
0
1
2
3
4
5*
.
0
1
2
3
4
5*
* 
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

kernel
bias*
�
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses

kernel
bias*
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses* 
�
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses

kernel
bias*
�
4
_variables
5_iterations
6_learning_rate
7_index_dict
8
_momentums
9_velocities
:_update_step_xla*
* 

;serving_default* 
]W
VARIABLE_VALUEcritic_network/dense_3/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEcritic_network/dense_3/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEcritic_network/dense_4/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEcritic_network/dense_4/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEcritic_network/dense_5/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEcritic_network/dense_5/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
	1

2
3*
* 
* 
* 
* 
* 

0
1*

0
1*
* 
�
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*

Atrace_0* 

Btrace_0* 

0
1*

0
1*
* 
�
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*

Htrace_0* 

Itrace_0* 
* 
* 
* 
�
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses* 

Otrace_0* 

Ptrace_0* 

0
1*

0
1*
* 
�
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*

Vtrace_0* 

Wtrace_0* 
b
50
X1
Y2
Z3
[4
\5
]6
^7
_8
`9
a10
b11
c12*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
X0
Z1
\2
^3
`4
b5*
.
Y0
[1
]2
_3
a4
c5*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
oi
VARIABLE_VALUE$Adam/m/critic_network/dense_3/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE$Adam/v/critic_network/dense_3/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/m/critic_network/dense_3/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/v/critic_network/dense_3/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE$Adam/m/critic_network/dense_4/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE$Adam/v/critic_network/dense_4/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/m/critic_network/dense_4/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/v/critic_network/dense_4/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE$Adam/m/critic_network/dense_5/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE$Adam/v/critic_network/dense_5/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/critic_network/dense_5/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/critic_network/dense_5/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamecritic_network/dense_3/kernelcritic_network/dense_3/biascritic_network/dense_4/kernelcritic_network/dense_4/biascritic_network/dense_5/kernelcritic_network/dense_5/bias	iterationlearning_rate$Adam/m/critic_network/dense_3/kernel$Adam/v/critic_network/dense_3/kernel"Adam/m/critic_network/dense_3/bias"Adam/v/critic_network/dense_3/bias$Adam/m/critic_network/dense_4/kernel$Adam/v/critic_network/dense_4/kernel"Adam/m/critic_network/dense_4/bias"Adam/v/critic_network/dense_4/bias$Adam/m/critic_network/dense_5/kernel$Adam/v/critic_network/dense_5/kernel"Adam/m/critic_network/dense_5/bias"Adam/v/critic_network/dense_5/biasConst*!
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_141090
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamecritic_network/dense_3/kernelcritic_network/dense_3/biascritic_network/dense_4/kernelcritic_network/dense_4/biascritic_network/dense_5/kernelcritic_network/dense_5/bias	iterationlearning_rate$Adam/m/critic_network/dense_3/kernel$Adam/v/critic_network/dense_3/kernel"Adam/m/critic_network/dense_3/bias"Adam/v/critic_network/dense_3/bias$Adam/m/critic_network/dense_4/kernel$Adam/v/critic_network/dense_4/kernel"Adam/m/critic_network/dense_4/bias"Adam/v/critic_network/dense_4/bias$Adam/m/critic_network/dense_5/kernel$Adam/v/critic_network/dense_5/kernel"Adam/m/critic_network/dense_5/bias"Adam/v/critic_network/dense_5/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_141159ц
��
�
__inference__traced_save_141090
file_prefixG
4read_disablecopyonread_critic_network_dense_3_kernel:	�C
4read_1_disablecopyonread_critic_network_dense_3_bias:	�J
6read_2_disablecopyonread_critic_network_dense_4_kernel:
��C
4read_3_disablecopyonread_critic_network_dense_4_bias:	�I
6read_4_disablecopyonread_critic_network_dense_5_kernel:	�B
4read_5_disablecopyonread_critic_network_dense_5_bias:,
"read_6_disablecopyonread_iteration:	 0
&read_7_disablecopyonread_learning_rate: P
=read_8_disablecopyonread_adam_m_critic_network_dense_3_kernel:	�P
=read_9_disablecopyonread_adam_v_critic_network_dense_3_kernel:	�K
<read_10_disablecopyonread_adam_m_critic_network_dense_3_bias:	�K
<read_11_disablecopyonread_adam_v_critic_network_dense_3_bias:	�R
>read_12_disablecopyonread_adam_m_critic_network_dense_4_kernel:
��R
>read_13_disablecopyonread_adam_v_critic_network_dense_4_kernel:
��K
<read_14_disablecopyonread_adam_m_critic_network_dense_4_bias:	�K
<read_15_disablecopyonread_adam_v_critic_network_dense_4_bias:	�Q
>read_16_disablecopyonread_adam_m_critic_network_dense_5_kernel:	�Q
>read_17_disablecopyonread_adam_v_critic_network_dense_5_kernel:	�J
<read_18_disablecopyonread_adam_m_critic_network_dense_5_bias:J
<read_19_disablecopyonread_adam_v_critic_network_dense_5_bias:
savev2_const
identity_41��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
Read/DisableCopyOnReadDisableCopyOnRead4read_disablecopyonread_critic_network_dense_3_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp4read_disablecopyonread_critic_network_dense_3_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0j
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�b

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_1/DisableCopyOnReadDisableCopyOnRead4read_1_disablecopyonread_critic_network_dense_3_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp4read_1_disablecopyonread_critic_network_dense_3_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_2/DisableCopyOnReadDisableCopyOnRead6read_2_disablecopyonread_critic_network_dense_4_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp6read_2_disablecopyonread_critic_network_dense_4_kernel^Read_2/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0o

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��e

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_3/DisableCopyOnReadDisableCopyOnRead4read_3_disablecopyonread_critic_network_dense_4_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp4read_3_disablecopyonread_critic_network_dense_4_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_4/DisableCopyOnReadDisableCopyOnRead6read_4_disablecopyonread_critic_network_dense_5_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp6read_4_disablecopyonread_critic_network_dense_5_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0n

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�d

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_5/DisableCopyOnReadDisableCopyOnRead4read_5_disablecopyonread_critic_network_dense_5_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp4read_5_disablecopyonread_critic_network_dense_5_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_6/DisableCopyOnReadDisableCopyOnRead"read_6_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp"read_6_disablecopyonread_iteration^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	f
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_7/DisableCopyOnReadDisableCopyOnRead&read_7_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp&read_7_disablecopyonread_learning_rate^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_8/DisableCopyOnReadDisableCopyOnRead=read_8_disablecopyonread_adam_m_critic_network_dense_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp=read_8_disablecopyonread_adam_m_critic_network_dense_3_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0o
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_9/DisableCopyOnReadDisableCopyOnRead=read_9_disablecopyonread_adam_v_critic_network_dense_3_kernel"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp=read_9_disablecopyonread_adam_v_critic_network_dense_3_kernel^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0o
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_10/DisableCopyOnReadDisableCopyOnRead<read_10_disablecopyonread_adam_m_critic_network_dense_3_bias"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp<read_10_disablecopyonread_adam_m_critic_network_dense_3_bias^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_11/DisableCopyOnReadDisableCopyOnRead<read_11_disablecopyonread_adam_v_critic_network_dense_3_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp<read_11_disablecopyonread_adam_v_critic_network_dense_3_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_12/DisableCopyOnReadDisableCopyOnRead>read_12_disablecopyonread_adam_m_critic_network_dense_4_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp>read_12_disablecopyonread_adam_m_critic_network_dense_4_kernel^Read_12/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_13/DisableCopyOnReadDisableCopyOnRead>read_13_disablecopyonread_adam_v_critic_network_dense_4_kernel"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp>read_13_disablecopyonread_adam_v_critic_network_dense_4_kernel^Read_13/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_14/DisableCopyOnReadDisableCopyOnRead<read_14_disablecopyonread_adam_m_critic_network_dense_4_bias"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp<read_14_disablecopyonread_adam_m_critic_network_dense_4_bias^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_15/DisableCopyOnReadDisableCopyOnRead<read_15_disablecopyonread_adam_v_critic_network_dense_4_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp<read_15_disablecopyonread_adam_v_critic_network_dense_4_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_16/DisableCopyOnReadDisableCopyOnRead>read_16_disablecopyonread_adam_m_critic_network_dense_5_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp>read_16_disablecopyonread_adam_m_critic_network_dense_5_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_17/DisableCopyOnReadDisableCopyOnRead>read_17_disablecopyonread_adam_v_critic_network_dense_5_kernel"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp>read_17_disablecopyonread_adam_v_critic_network_dense_5_kernel^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_18/DisableCopyOnReadDisableCopyOnRead<read_18_disablecopyonread_adam_m_critic_network_dense_5_bias"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp<read_18_disablecopyonread_adam_m_critic_network_dense_5_bias^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_19/DisableCopyOnReadDisableCopyOnRead<read_19_disablecopyonread_adam_v_critic_network_dense_5_bias"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp<read_19_disablecopyonread_adam_v_critic_network_dense_5_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *#
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_40Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_41IdentityIdentity_40:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_41Identity_41:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,: : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:=9
7
_user_specified_namecritic_network/dense_3/kernel:;7
5
_user_specified_namecritic_network/dense_3/bias:=9
7
_user_specified_namecritic_network/dense_4/kernel:;7
5
_user_specified_namecritic_network/dense_4/bias:=9
7
_user_specified_namecritic_network/dense_5/kernel:;7
5
_user_specified_namecritic_network/dense_5/bias:)%
#
_user_specified_name	iteration:-)
'
_user_specified_namelearning_rate:D	@
>
_user_specified_name&$Adam/m/critic_network/dense_3/kernel:D
@
>
_user_specified_name&$Adam/v/critic_network/dense_3/kernel:B>
<
_user_specified_name$"Adam/m/critic_network/dense_3/bias:B>
<
_user_specified_name$"Adam/v/critic_network/dense_3/bias:D@
>
_user_specified_name&$Adam/m/critic_network/dense_4/kernel:D@
>
_user_specified_name&$Adam/v/critic_network/dense_4/kernel:B>
<
_user_specified_name$"Adam/m/critic_network/dense_4/bias:B>
<
_user_specified_name$"Adam/v/critic_network/dense_4/bias:D@
>
_user_specified_name&$Adam/m/critic_network/dense_5/kernel:D@
>
_user_specified_name&$Adam/v/critic_network/dense_5/kernel:B>
<
_user_specified_name$"Adam/m/critic_network/dense_5/bias:B>
<
_user_specified_name$"Adam/v/critic_network/dense_5/bias:=9

_output_shapes
: 

_user_specified_nameConst
�

�
C__inference_dense_4_layer_call_and_return_conditional_losses_140767

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�

�
$__inference_signature_wrapper_140858
input_1
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_140711o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1:&"
 
_user_specified_name140844:&"
 
_user_specified_name140846:&"
 
_user_specified_name140848:&"
 
_user_specified_name140850:&"
 
_user_specified_name140852:&"
 
_user_specified_name140854
�A
�
!__inference__wrapped_model_140711
input_1K
8critic_network_dense_3_tensordot_readvariableop_resource:	�E
6critic_network_dense_3_biasadd_readvariableop_resource:	�I
5critic_network_dense_4_matmul_readvariableop_resource:
��E
6critic_network_dense_4_biasadd_readvariableop_resource:	�H
5critic_network_dense_5_matmul_readvariableop_resource:	�D
6critic_network_dense_5_biasadd_readvariableop_resource:
identity��-critic_network/dense_3/BiasAdd/ReadVariableOp�/critic_network/dense_3/Tensordot/ReadVariableOp�-critic_network/dense_4/BiasAdd/ReadVariableOp�,critic_network/dense_4/MatMul/ReadVariableOp�-critic_network/dense_5/BiasAdd/ReadVariableOp�,critic_network/dense_5/MatMul/ReadVariableOp�
/critic_network/dense_3/Tensordot/ReadVariableOpReadVariableOp8critic_network_dense_3_tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0o
%critic_network/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:z
%critic_network/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          k
&critic_network/dense_3/Tensordot/ShapeShapeinput_1*
T0*
_output_shapes
::��p
.critic_network/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
)critic_network/dense_3/Tensordot/GatherV2GatherV2/critic_network/dense_3/Tensordot/Shape:output:0.critic_network/dense_3/Tensordot/free:output:07critic_network/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:r
0critic_network/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
+critic_network/dense_3/Tensordot/GatherV2_1GatherV2/critic_network/dense_3/Tensordot/Shape:output:0.critic_network/dense_3/Tensordot/axes:output:09critic_network/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
&critic_network/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
%critic_network/dense_3/Tensordot/ProdProd2critic_network/dense_3/Tensordot/GatherV2:output:0/critic_network/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: r
(critic_network/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
'critic_network/dense_3/Tensordot/Prod_1Prod4critic_network/dense_3/Tensordot/GatherV2_1:output:01critic_network/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: n
,critic_network/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'critic_network/dense_3/Tensordot/concatConcatV2.critic_network/dense_3/Tensordot/free:output:0.critic_network/dense_3/Tensordot/axes:output:05critic_network/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
&critic_network/dense_3/Tensordot/stackPack.critic_network/dense_3/Tensordot/Prod:output:00critic_network/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
*critic_network/dense_3/Tensordot/transpose	Transposeinput_10critic_network/dense_3/Tensordot/concat:output:0*
T0*/
_output_shapes
:����������
(critic_network/dense_3/Tensordot/ReshapeReshape.critic_network/dense_3/Tensordot/transpose:y:0/critic_network/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
'critic_network/dense_3/Tensordot/MatMulMatMul1critic_network/dense_3/Tensordot/Reshape:output:07critic_network/dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
(critic_network/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�p
.critic_network/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
)critic_network/dense_3/Tensordot/concat_1ConcatV22critic_network/dense_3/Tensordot/GatherV2:output:01critic_network/dense_3/Tensordot/Const_2:output:07critic_network/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
 critic_network/dense_3/TensordotReshape1critic_network/dense_3/Tensordot/MatMul:product:02critic_network/dense_3/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:�����������
-critic_network/dense_3/BiasAdd/ReadVariableOpReadVariableOp6critic_network_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
critic_network/dense_3/BiasAddBiasAdd)critic_network/dense_3/Tensordot:output:05critic_network/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
critic_network/dense_3/ReluRelu'critic_network/dense_3/BiasAdd:output:0*
T0*0
_output_shapes
:����������o
critic_network/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   �
 critic_network/flatten_1/ReshapeReshape)critic_network/dense_3/Relu:activations:0'critic_network/flatten_1/Const:output:0*
T0*(
_output_shapes
:�����������
,critic_network/dense_4/MatMul/ReadVariableOpReadVariableOp5critic_network_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
critic_network/dense_4/MatMulMatMul)critic_network/flatten_1/Reshape:output:04critic_network/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-critic_network/dense_4/BiasAdd/ReadVariableOpReadVariableOp6critic_network_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
critic_network/dense_4/BiasAddBiasAdd'critic_network/dense_4/MatMul:product:05critic_network/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
critic_network/dense_4/ReluRelu'critic_network/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
,critic_network/dense_5/MatMul/ReadVariableOpReadVariableOp5critic_network_dense_5_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
critic_network/dense_5/MatMulMatMul)critic_network/dense_4/Relu:activations:04critic_network/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-critic_network/dense_5/BiasAdd/ReadVariableOpReadVariableOp6critic_network_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
critic_network/dense_5/BiasAddBiasAdd'critic_network/dense_5/MatMul:product:05critic_network/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������v
IdentityIdentity'critic_network/dense_5/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp.^critic_network/dense_3/BiasAdd/ReadVariableOp0^critic_network/dense_3/Tensordot/ReadVariableOp.^critic_network/dense_4/BiasAdd/ReadVariableOp-^critic_network/dense_4/MatMul/ReadVariableOp.^critic_network/dense_5/BiasAdd/ReadVariableOp-^critic_network/dense_5/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 2^
-critic_network/dense_3/BiasAdd/ReadVariableOp-critic_network/dense_3/BiasAdd/ReadVariableOp2b
/critic_network/dense_3/Tensordot/ReadVariableOp/critic_network/dense_3/Tensordot/ReadVariableOp2^
-critic_network/dense_4/BiasAdd/ReadVariableOp-critic_network/dense_4/BiasAdd/ReadVariableOp2\
,critic_network/dense_4/MatMul/ReadVariableOp,critic_network/dense_4/MatMul/ReadVariableOp2^
-critic_network/dense_5/BiasAdd/ReadVariableOp-critic_network/dense_5/BiasAdd/ReadVariableOp2\
,critic_network/dense_5/MatMul/ReadVariableOp,critic_network/dense_5/MatMul/ReadVariableOp:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�	
�
C__inference_dense_5_layer_call_and_return_conditional_losses_140782

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_140929

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
J__inference_critic_network_layer_call_and_return_conditional_losses_140789
input_1!
dense_3_140745:	�
dense_3_140747:	�"
dense_4_140768:
��
dense_4_140770:	�!
dense_5_140783:	�
dense_5_140785:
identity��dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_3_140745dense_3_140747*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_140744�
flatten_1/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_140755�
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_4_140768dense_4_140770*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_140767�
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_140783dense_5_140785*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_140782w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1:&"
 
_user_specified_name140745:&"
 
_user_specified_name140747:&"
 
_user_specified_name140768:&"
 
_user_specified_name140770:&"
 
_user_specified_name140783:&"
 
_user_specified_name140785
�
�
C__inference_dense_3_layer_call_and_return_conditional_losses_140744

inputs4
!tensordot_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:c
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:}
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*/
_output_shapes
:����������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*0
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
(__inference_dense_3_layer_call_fn_140867

inputs
unknown:	�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_140744x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:&"
 
_user_specified_name140861:&"
 
_user_specified_name140863
�e
�
"__inference__traced_restore_141159
file_prefixA
.assignvariableop_critic_network_dense_3_kernel:	�=
.assignvariableop_1_critic_network_dense_3_bias:	�D
0assignvariableop_2_critic_network_dense_4_kernel:
��=
.assignvariableop_3_critic_network_dense_4_bias:	�C
0assignvariableop_4_critic_network_dense_5_kernel:	�<
.assignvariableop_5_critic_network_dense_5_bias:&
assignvariableop_6_iteration:	 *
 assignvariableop_7_learning_rate: J
7assignvariableop_8_adam_m_critic_network_dense_3_kernel:	�J
7assignvariableop_9_adam_v_critic_network_dense_3_kernel:	�E
6assignvariableop_10_adam_m_critic_network_dense_3_bias:	�E
6assignvariableop_11_adam_v_critic_network_dense_3_bias:	�L
8assignvariableop_12_adam_m_critic_network_dense_4_kernel:
��L
8assignvariableop_13_adam_v_critic_network_dense_4_kernel:
��E
6assignvariableop_14_adam_m_critic_network_dense_4_bias:	�E
6assignvariableop_15_adam_v_critic_network_dense_4_bias:	�K
8assignvariableop_16_adam_m_critic_network_dense_5_kernel:	�K
8assignvariableop_17_adam_v_critic_network_dense_5_kernel:	�D
6assignvariableop_18_adam_m_critic_network_dense_5_bias:D
6assignvariableop_19_adam_v_critic_network_dense_5_bias:
identity_21��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*h
_output_shapesV
T:::::::::::::::::::::*#
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp.assignvariableop_critic_network_dense_3_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp.assignvariableop_1_critic_network_dense_3_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp0assignvariableop_2_critic_network_dense_4_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp.assignvariableop_3_critic_network_dense_4_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp0assignvariableop_4_critic_network_dense_5_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp.assignvariableop_5_critic_network_dense_5_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_iterationIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_learning_rateIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp7assignvariableop_8_adam_m_critic_network_dense_3_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp7assignvariableop_9_adam_v_critic_network_dense_3_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp6assignvariableop_10_adam_m_critic_network_dense_3_biasIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp6assignvariableop_11_adam_v_critic_network_dense_3_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp8assignvariableop_12_adam_m_critic_network_dense_4_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp8assignvariableop_13_adam_v_critic_network_dense_4_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp6assignvariableop_14_adam_m_critic_network_dense_4_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp6assignvariableop_15_adam_v_critic_network_dense_4_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp8assignvariableop_16_adam_m_critic_network_dense_5_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp8assignvariableop_17_adam_v_critic_network_dense_5_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp6assignvariableop_18_adam_m_critic_network_dense_5_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp6assignvariableop_19_adam_v_critic_network_dense_5_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_20Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_21IdentityIdentity_20:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_21Identity_21:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*: : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:=9
7
_user_specified_namecritic_network/dense_3/kernel:;7
5
_user_specified_namecritic_network/dense_3/bias:=9
7
_user_specified_namecritic_network/dense_4/kernel:;7
5
_user_specified_namecritic_network/dense_4/bias:=9
7
_user_specified_namecritic_network/dense_5/kernel:;7
5
_user_specified_namecritic_network/dense_5/bias:)%
#
_user_specified_name	iteration:-)
'
_user_specified_namelearning_rate:D	@
>
_user_specified_name&$Adam/m/critic_network/dense_3/kernel:D
@
>
_user_specified_name&$Adam/v/critic_network/dense_3/kernel:B>
<
_user_specified_name$"Adam/m/critic_network/dense_3/bias:B>
<
_user_specified_name$"Adam/v/critic_network/dense_3/bias:D@
>
_user_specified_name&$Adam/m/critic_network/dense_4/kernel:D@
>
_user_specified_name&$Adam/v/critic_network/dense_4/kernel:B>
<
_user_specified_name$"Adam/m/critic_network/dense_4/bias:B>
<
_user_specified_name$"Adam/v/critic_network/dense_4/bias:D@
>
_user_specified_name&$Adam/m/critic_network/dense_5/kernel:D@
>
_user_specified_name&$Adam/v/critic_network/dense_5/kernel:B>
<
_user_specified_name$"Adam/m/critic_network/dense_5/bias:B>
<
_user_specified_name$"Adam/v/critic_network/dense_5/bias
�	
�
C__inference_dense_5_layer_call_and_return_conditional_losses_140948

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�

�
C__inference_dense_4_layer_call_and_return_conditional_losses_140918

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
�
C__inference_dense_3_layer_call_and_return_conditional_losses_140898

inputs4
!tensordot_readvariableop_resource:	�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp{
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:c
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:}
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*/
_output_shapes
:����������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*0
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
�
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_140755

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
/__inference_critic_network_layer_call_fn_140806
input_1
unknown:	�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_critic_network_layer_call_and_return_conditional_losses_140789o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1:&"
 
_user_specified_name140792:&"
 
_user_specified_name140794:&"
 
_user_specified_name140796:&"
 
_user_specified_name140798:&"
 
_user_specified_name140800:&"
 
_user_specified_name140802
�
�
(__inference_dense_5_layer_call_fn_140938

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_140782o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:&"
 
_user_specified_name140932:&"
 
_user_specified_name140934
�
F
*__inference_flatten_1_layer_call_fn_140923

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_140755a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_dense_4_layer_call_fn_140907

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_140767p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:&"
 
_user_specified_name140901:&"
 
_user_specified_name140903"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_18
serving_default_input_1:0���������<
output_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:�d
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
fc1
	fc2

flatten
q
	optimizer
loss

signatures"
_tf_keras_model
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
trace_02�
/__inference_critic_network_layer_call_fn_140806�
���
FullArgSpec
args�	
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0
�
trace_02�
J__inference_critic_network_layer_call_and_return_conditional_losses_140789�
���
FullArgSpec
args�	
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0
�B�
!__inference__wrapped_model_140711input_1"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses"
_tf_keras_layer
�
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
4
_variables
5_iterations
6_learning_rate
7_index_dict
8
_momentums
9_velocities
:_update_step_xla"
experimentalOptimizer
 "
trackable_dict_wrapper
,
;serving_default"
signature_map
0:.	�2critic_network/dense_3/kernel
*:(�2critic_network/dense_3/bias
1:/
��2critic_network/dense_4/kernel
*:(�2critic_network/dense_4/bias
0:.	�2critic_network/dense_5/kernel
):'2critic_network/dense_5/bias
 "
trackable_list_wrapper
<
0
	1

2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
/__inference_critic_network_layer_call_fn_140806input_1"�
���
FullArgSpec
args�	
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_critic_network_layer_call_and_return_conditional_losses_140789input_1"�
���
FullArgSpec
args�	
jstate
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
�
Atrace_02�
(__inference_dense_3_layer_call_fn_140867�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zAtrace_0
�
Btrace_02�
C__inference_dense_3_layer_call_and_return_conditional_losses_140898�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zBtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
�
Htrace_02�
(__inference_dense_4_layer_call_fn_140907�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zHtrace_0
�
Itrace_02�
C__inference_dense_4_layer_call_and_return_conditional_losses_140918�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zItrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
�
Otrace_02�
*__inference_flatten_1_layer_call_fn_140923�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zOtrace_0
�
Ptrace_02�
E__inference_flatten_1_layer_call_and_return_conditional_losses_140929�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zPtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Qnon_trainable_variables

Rlayers
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
�
Vtrace_02�
(__inference_dense_5_layer_call_fn_140938�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zVtrace_0
�
Wtrace_02�
C__inference_dense_5_layer_call_and_return_conditional_losses_140948�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zWtrace_0
~
50
X1
Y2
Z3
[4
\5
]6
^7
_8
`9
a10
b11
c12"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
J
X0
Z1
\2
^3
`4
b5"
trackable_list_wrapper
J
Y0
[1
]2
_3
a4
c5"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
$__inference_signature_wrapper_140858input_1"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs�
	jinput_1
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_3_layer_call_fn_140867inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_3_layer_call_and_return_conditional_losses_140898inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_4_layer_call_fn_140907inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_4_layer_call_and_return_conditional_losses_140918inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_flatten_1_layer_call_fn_140923inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_flatten_1_layer_call_and_return_conditional_losses_140929inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_dense_5_layer_call_fn_140938inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_5_layer_call_and_return_conditional_losses_140948inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
5:3	�2$Adam/m/critic_network/dense_3/kernel
5:3	�2$Adam/v/critic_network/dense_3/kernel
/:-�2"Adam/m/critic_network/dense_3/bias
/:-�2"Adam/v/critic_network/dense_3/bias
6:4
��2$Adam/m/critic_network/dense_4/kernel
6:4
��2$Adam/v/critic_network/dense_4/kernel
/:-�2"Adam/m/critic_network/dense_4/bias
/:-�2"Adam/v/critic_network/dense_4/bias
5:3	�2$Adam/m/critic_network/dense_5/kernel
5:3	�2$Adam/v/critic_network/dense_5/kernel
.:,2"Adam/m/critic_network/dense_5/bias
.:,2"Adam/v/critic_network/dense_5/bias�
!__inference__wrapped_model_140711w8�5
.�+
)�&
input_1���������
� "3�0
.
output_1"�
output_1����������
J__inference_critic_network_layer_call_and_return_conditional_losses_140789p8�5
.�+
)�&
input_1���������
� ",�)
"�
tensor_0���������
� �
/__inference_critic_network_layer_call_fn_140806e8�5
.�+
)�&
input_1���������
� "!�
unknown����������
C__inference_dense_3_layer_call_and_return_conditional_losses_140898t7�4
-�*
(�%
inputs���������
� "5�2
+�(
tensor_0����������
� �
(__inference_dense_3_layer_call_fn_140867i7�4
-�*
(�%
inputs���������
� "*�'
unknown�����������
C__inference_dense_4_layer_call_and_return_conditional_losses_140918e0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
(__inference_dense_4_layer_call_fn_140907Z0�-
&�#
!�
inputs����������
� ""�
unknown�����������
C__inference_dense_5_layer_call_and_return_conditional_losses_140948d0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������
� �
(__inference_dense_5_layer_call_fn_140938Y0�-
&�#
!�
inputs����������
� "!�
unknown����������
E__inference_flatten_1_layer_call_and_return_conditional_losses_140929i8�5
.�+
)�&
inputs����������
� "-�*
#� 
tensor_0����������
� �
*__inference_flatten_1_layer_call_fn_140923^8�5
.�+
)�&
inputs����������
� ""�
unknown�����������
$__inference_signature_wrapper_140858�C�@
� 
9�6
4
input_1)�&
input_1���������"3�0
.
output_1"�
output_1���������