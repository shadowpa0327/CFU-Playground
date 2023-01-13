// Copyright 2021 The CFU-Playground Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


`define BUFFER_A_DEPTH  10 
`define BUFFER_B_DEPTH  10
`define BUFFER_C_DEPTH  2 


module Cfu (
  input               cmd_valid,
  output              cmd_ready,
  input      [9:0]    cmd_payload_function_id,
  input      [31:0]   cmd_payload_inputs_0,
  input      [31:0]   cmd_payload_inputs_1,
  output              rsp_valid,
  input               rsp_ready,
  output     [31:0]   rsp_payload_outputs_0,
  input               reset,
  input               clk
);


  // decoder related signal 

  wire       [6:0]    funct_7;
  wire       [2:0]    funct_3;
  assign funct_7 = (cmd_valid)? cmd_payload_function_id[9:3] : 0;
  assign funct_3 = (cmd_valid)? cmd_payload_function_id[2:0] : 0;
  // global signal  
  wire rst_n;
  assign rst_n = ~reset;

  // signal for global buffer 
  wire                A_wr_en;
  wire       [`BUFFER_A_DEPTH-1:0]   A_index_TPU;
  wire       [`BUFFER_A_DEPTH-1:0]   A_index;
  wire       [7:0]   A_data_in;
  wire       [7:0]   A_data_out;

  wire                B_wr_en;
  wire       [`BUFFER_B_DEPTH-1:0]   B_index_TPU;
  wire       [`BUFFER_B_DEPTH-1:0]   B_index;
  wire       [31:0]   B_data_in;
  wire       [31:0]   B_data_out;

  wire                C_wr_en;
  wire       [`BUFFER_C_DEPTH-1:0]   C_index_TPU;
  wire       [`BUFFER_C_DEPTH-1:0]   C_index;
  wire       [127:0]  C_data_in;
  wire       [127:0]  C_data_out;

  // signal of TPU 
  wire            in_valid;
  wire [9:0]      K;
  wire [9:0]      M;
  wire [9:0]      N;
  wire            busy;

  global_buffer #(
    .ADDR_BITS(`BUFFER_A_DEPTH),
    .DATA_BITS(8)
  )
  gbuff_A(
      .clk(clk),
      .rst_n(rst_n),
      .wr_en(A_wr_en),
      .index(A_index),
      .data_in(A_data_in),
      .data_out(A_data_out)
  );

  global_buffer #(
      .ADDR_BITS(`BUFFER_B_DEPTH),
      .DATA_BITS(32)
  ) gbuff_B(
      .clk(clk),
      .rst_n(rst_n),
      .wr_en(B_wr_en),
      .index(B_index),
      .data_in(B_data_in),
      .data_out(B_data_out)
  );

  global_buffer #(
      .ADDR_BITS(`BUFFER_C_DEPTH),
      .DATA_BITS(128)
  ) gbuff_C(
      .clk(clk),
      .rst_n(rst_n),
      .wr_en(C_wr_en),
      .index(C_index),
      .data_in(C_data_in),
      .data_out(C_data_out)
  );
  
  TPU My_TPU(
      .clk            (clk),     
      .rst_n          (rst_n),     
      .in_valid       (in_valid),         
      .K              (K), 
      .M              (M), 
      .N              (N), 
      .busy           (busy),     
      .A_wr_en        (A_wr_en),         
      .A_index        (A_index_TPU),         
      .A_data_in      (A_data_in),         
      .A_data_out     (A_data_out),         
      .B_wr_en        (B_wr_en),         
      .B_index        (B_index_TPU),         
      .B_data_in      (B_data_in),         
      .B_data_out     (B_data_out),         
      .C_wr_en        (C_wr_en),         
      .C_index        (C_index_TPU),         
      .C_data_in      (C_data_in),         
      .C_data_out     (C_data_out)         
  );

  // signal controller


  // FSM 
  // WA : Write to buffer A,  
  // WB : Write to buffer B
  // READ : Read Buffer
  parameter IDLE = 0,CAL_PREPARE =1, CAL = 6, WA = 2, WB = 3, READ = 4, FINISH = 5; 
  
  reg [2:0] state;
  reg [2:0] next_state;

  always@(posedge clk, negedge rst_n)begin
    if(!rst_n)begin
      state <= IDLE;
    end
    else begin
      state <= next_state;
    end
  end

  always @(*)begin
    case (state)
      IDLE: next_state = (!cmd_valid)? IDLE :(funct_7 == 1)? FINISH : (funct_7 == 2) ? FINISH : (funct_7 == 3) ? CAL_PREPARE : (funct_7 == 4) ? FINISH : IDLE;
      WA  : next_state = FINISH;
      WB  : next_state = FINISH;
      READ  : next_state = FINISH; 
      CAL_PREPARE : next_state = CAL;
      CAL : next_state = (busy) ? CAL : FINISH; 
      FINISH : next_state =  IDLE;
    endcase
  end

  // store input signal 
  wire [31:0] cmd_payload_inputs_0_comb;
  wire [31:0] cmd_payload_inputs_1_comb;

  assign cmd_payload_inputs_0_comb = (cmd_valid)? cmd_payload_inputs_0: 0;
  assign cmd_payload_inputs_1_comb = (cmd_valid)? cmd_payload_inputs_1: 0;


  // Control signal of buffer
  assign A_wr_en = (!cmd_valid)? 0 : (funct_7 == 1) ? 1 :0;
  assign B_wr_en = (!cmd_valid)? 0 : (funct_7 == 2) ? 1 :0;

  // data for buffer
  assign A_index   = (state == CAL) ? A_index_TPU : cmd_payload_inputs_0_comb[`BUFFER_A_DEPTH-1:0];
  assign A_data_in = cmd_payload_inputs_1_comb[31:24];
  assign B_index   = (state == CAL) ? B_index_TPU : cmd_payload_inputs_0_comb[`BUFFER_B_DEPTH-1:0];
  assign B_data_in = cmd_payload_inputs_1_comb;
  
  assign C_index   = (state == CAL) ? C_index_TPU : cmd_payload_inputs_0_comb[`BUFFER_C_DEPTH-1:0];

  // Control signal of TPU
  assign in_valid = (state == CAL_PREPARE) ? 1 : 0;
  assign M = 1;
  assign K = cmd_payload_inputs_0_comb[9:0];
  assign N = cmd_payload_inputs_1_comb[9:0];





  // Trivial handshaking for a combinational CFU
  assign rsp_valid = (state == FINISH) ? 1: 0;
  //assign rsp_valid = cmd_valid;
  //assign cmd_ready = ~rsp_valid;
  assign cmd_ready = rsp_valid;

  
  // prepera the result for the READ stage

  wire [31:0]  C_buffer_query_result;
  wire [1:0]   which_value_in_a_row;
  assign which_value_in_a_row = cmd_payload_inputs_1_comb[1:0];
  assign C_buffer_query_result = (which_value_in_a_row == 0)? C_data_out[127:96] : 
                                 (which_value_in_a_row == 1)? C_data_out[95:64]  : 
                                 (which_value_in_a_row == 2)? C_data_out[63:32]  : C_data_out[31:0];


  assign rsp_payload_outputs_0 = C_buffer_query_result;

endmodule


module global_buffer #(parameter ADDR_BITS=8, parameter DATA_BITS=8)(clk, rst_n, wr_en, index, data_in, data_out);

  input clk;
  input rst_n;
  input wr_en; // Write enable: 1->write 0->read
  input      [ADDR_BITS-1:0] index;
  input      [DATA_BITS-1:0]       data_in;
  output reg [DATA_BITS-1:0]       data_out;

  integer i;

  parameter DEPTH = 2**ADDR_BITS;

//----------------------------------------------------------------------------//
// Global buffer (Don't change the name)                                      //
//----------------------------------------------------------------------------//
  // reg [`GBUFF_ADDR_SIZE-1:0] gbuff [`WORD_SIZE-1:0];
  (* ram_style = "block"*)reg [DATA_BITS-1:0] gbuff [DEPTH-1:0];

//----------------------------------------------------------------------------//
// Global buffer read write behavior                                          //
//----------------------------------------------------------------------------//
  always @ (negedge clk) begin
    if(wr_en) begin
      gbuff[index] <= data_in;
    end
    else begin
      data_out <= gbuff[index];
    end
  end

endmodule



// Student 0811514, custom TPU design
module TPU(
    clk,
    rst_n,

    in_valid,
    K,
    M,
    N,
    busy,

    A_wr_en,
    A_index,
    A_data_in,
    A_data_out,

    B_wr_en,
    B_index,
    B_data_in,
    B_data_out,

    C_wr_en,
    C_index,
    C_data_in,
    C_data_out
);



input clk;
input rst_n;
input            in_valid;
input [9:0]      K;
input [9:0]      M;
input [9:0]      N;
output           busy;

output           A_wr_en;
output [`BUFFER_A_DEPTH-1:0]    A_index;
output [7:0]    A_data_in;
input  [7:0]    A_data_out;

output           B_wr_en;

output [`BUFFER_B_DEPTH-1:0]    B_index; // (V)

output [31:0]    B_data_in; 
input  [31:0]    B_data_out; // (V)

output           C_wr_en;  // (V)
output [`BUFFER_C_DEPTH-1:0]    C_index;  // (V)
output [127:0]   C_data_in;  // (V)
input  [127:0]   C_data_out; // (X)

//* Implement your design here
// state
parameter IDLE = 0,CAL =1, WB = 2, PRE_LOAD=3;
reg [2:0] next;
reg  [2:0] state;
// FSM control data
reg   [9:0] K_ff, M_ff, N_ff;
wire  [9:0] K_comb, M_comb, N_comb;
reg   [9:0] counterK_ff;  
wire  [9:0] counterK_comb;  
//logic [7:0] counterA_comb,counterA_ff,counterB_comb,counterB_ff;
reg   [2:0] counterWB_ff;
wire  [2:0] counterWB_comb;  
// 
wire  cal_invalid;
reg  [`BUFFER_A_DEPTH-1:0] A_index_ff; 
reg  [`BUFFER_B_DEPTH-1:0] B_index_ff; 
reg  [`BUFFER_C_DEPTH-1:0] C_index_ff;

wire [`BUFFER_A_DEPTH-1:0] A_index_comb;
wire [`BUFFER_B_DEPTH-1:0] B_index_comb;
wire [`BUFFER_C_DEPTH-1:0] C_index_comb;

wire [7:0] A_data_input;
wire [31:0] B_data_input;

reg  [7:0] A_data_input_ff;
reg  [31:0] B_data_input_ff;


// FSM
always @(posedge clk,negedge rst_n)begin
	if(!rst_n)begin
		state <= IDLE;
	end
	else begin
		state <= next;
	end
end

always @(*)begin
	case (state)
		IDLE: next = (in_valid)? CAL: IDLE;
    PRE_LOAD :next = CAL;
		CAL : next = (counterK_ff == K_ff+3)? WB : CAL;		
		WB: begin 
		  if(counterWB_ff==1) begin 
          next = IDLE;
          end
		  else begin
		      next = WB;
		  end
        end
        default: next = IDLE;
	endcase
end
// busy singal
assign busy = (state==IDLE)? 0 : 1;

// store M N K
always @(posedge clk,negedge rst_n)begin
	if(!rst_n)begin
		N_ff <= 0;
		M_ff <= 0;
		K_ff <= 0;		
	end
	else begin
		N_ff <= N_comb;
		M_ff <= M_comb;
		K_ff <= K_comb;		
	end
end
assign N_comb = (in_valid)? N : N_ff;
assign M_comb = (in_valid)? M : M_ff;
assign K_comb = (in_valid)? K : K_ff;


// update counter,index
always @(posedge clk,negedge rst_n)begin
	if(!rst_n)begin
		B_index_ff<= 0;
		A_index_ff<= 0;
		C_index_ff<=0;
		counterK_ff<=0;
		counterWB_ff<=0;
	end
	else begin
		B_index_ff<= B_index_comb;
		A_index_ff<= A_index_comb;
		C_index_ff<=C_index_comb;
		counterK_ff<=counterK_comb;
		counterWB_ff<=counterWB_comb;	
	end
end

assign cal_invalid = (state==CAL && counterK_ff < K_ff)? 1 : 0;
assign A_data_input = (clear_TPU==1)? 0 : (cal_invalid || state==PRE_LOAD)?  A_data_out : 0;
assign B_data_input = (clear_TPU==1)? 0 : (cal_invalid || state==PRE_LOAD)?  B_data_out : 0;

assign B_index_comb = (state==IDLE)? 0 : (cal_invalid)? B_index_ff+1 : (counterWB_ff==1) ? 0 : B_index_ff ;

assign A_index_comb = (state==IDLE)? 0 : (cal_invalid)? A_index_ff+1 : (counterWB_ff==1) ? 0 : A_index_ff ;

assign C_index_comb = (state==IDLE)? 0 : (state==WB)? C_index_ff+1 : C_index_ff ;

assign counterK_comb = (state==CAL)? counterK_ff+1 : 0;
assign counterWB_comb = (state==WB)? counterWB_ff+1 : 0;

assign A_index = A_index_ff;
assign B_index = B_index_ff;
reg   [7:0] A1_ff,  B1_ff;
wire  [7:0] A1_out, B1_out;
reg   [7:0] A2_ff[1:0],  B2_ff[1:0];
wire  [7:0] A2_out[1:0], B2_out[1:0];
reg   [7:0] A3_ff[2:0],  B3_ff[2:0];
wire  [7:0] A3_out[2:0], B3_out[2:0];

always @(posedge clk, negedge rst_n) begin
  if(!rst_n) begin
    A_data_input_ff <= 0;
    B_data_input_ff <= 0;
  end
  else begin
    A_data_input_ff <= A_data_input;
    B_data_input_ff <= B_data_input;
  end
end


always @(posedge clk,negedge rst_n) begin
    if(!rst_n) begin
        A1_ff <= 0;
        A2_ff[0] <= 0;
        A2_ff[1] <= 0;	
        A3_ff[0] <= 0;
        A3_ff[1] <= 0;
        A3_ff[2] <= 0;
        B1_ff <= 0;
        B2_ff[0] <= 0;
        B2_ff[1] <= 0;	
        B3_ff[0] <= 0;
        B3_ff[1] <= 0;
        B3_ff[2] <= 0;
    end
    else begin
        //A1_ff    <= A_data_input[23:16];
        //A2_ff[0] <= A_data_input[15:8];
        //A2_ff[1] <= A2_out[0];
        //A3_ff[0] <= A_data_input[7:0];
        //A3_ff[1] <= A3_out[0];
        //A3_ff[2] <= A3_out[1];

        B1_ff    <= B_data_input_ff[23:16];
        B2_ff[0] <= B_data_input_ff[15:8];
        B2_ff[1] <= B2_out[0];
        B3_ff[0] <= B_data_input_ff[7:0];
        B3_ff[1] <= B3_out[0];
        B3_ff[2] <= B3_out[1];
    end
end

// assign A1_out = A1_ff;
// assign A2_out[0] = A2_ff[0];
// assign A2_out[1] = A2_ff[1];
// assign A3_out[0] = A3_ff[0];
// assign A3_out[1] = A3_ff[1];
// assign A3_out[2] = A3_ff[2];


assign A1_out = 0;
assign A2_out[0] = 0;
assign A2_out[1] = 0;
assign A3_out[0] = 0;
assign A3_out[1] = 0;
assign A3_out[2] = 0;

assign B1_out = B1_ff;
assign B2_out[0] = B2_ff[0];
assign B2_out[1] = B2_ff[1];
assign B3_out[0] = B3_ff[0];
assign B3_out[1] = B3_ff[1];
assign B3_out[2] = B3_ff[2];


wire  [127:0] output_wire[3:0];
reg   [127:0] C_data_input_ff[3:0];
wire  [127:0] C_data_input_comb[3:0];
wire  clear_TPU;


assign clear_TPU = (counterWB_ff == 1)? 1 : 0;
// input to Module
SystolicArray AWDS2588(
	clk, rst_n, clear_TPU, 
	A_data_input_ff, A1_out, A2_out[1], A3_out[2], 
	B_data_input_ff[31:24], B1_out, B2_out[1], B3_out[2], 
	output_wire[0], output_wire[1], output_wire[2], output_wire[3]   
);


always @(posedge clk,negedge rst_n) begin
    if(!rst_n) begin
        C_data_input_ff[0] <= 0;
        C_data_input_ff[1] <= 0;
        C_data_input_ff[2] <= 0;
        C_data_input_ff[3] <= 0;
    end
    else begin
        C_data_input_ff[0] <= C_data_input_comb[0];
        C_data_input_ff[1] <= C_data_input_comb[1];
        C_data_input_ff[2] <= C_data_input_comb[2];
        C_data_input_ff[3] <= C_data_input_comb[3];
    end
end

assign C_data_input_comb[0] = (counterK_ff == K_ff+3)? output_wire[0]  :  C_data_input_ff[0];
assign C_data_input_comb[1] = (counterK_ff == K_ff+3)? output_wire[1]  : C_data_input_ff[1];
assign C_data_input_comb[2] = (counterK_ff == K_ff+3)? output_wire[2]  :  C_data_input_ff[2];
assign C_data_input_comb[3] = (counterK_ff == K_ff+3)? output_wire[3]  : C_data_input_ff[3];

assign C_wr_en = (state==WB)? (counterWB_ff<M_ff)? 1 : 0 : 0;
assign C_index = C_index_ff;
assign C_data_in = C_data_input_ff[counterWB_ff];

endmodule


module PE(inp_north, inp_west, clk, rst_n, clear_TPU, outp_south, outp_east, result);
	input [8-1:0] inp_north, inp_west;
	reg [8-1:0] outp_south_ff, outp_east_ff;
	output wire [8-1:0] outp_south, outp_east;
	input clk, rst_n;
	reg signed [32-1:0] result_ff;
	output wire signed [32-1:0] result;
	wire signed [32-1:0] multi;
	input clear_TPU;
	always @(negedge rst_n or posedge clk) begin
		if(!rst_n) begin
			result_ff <= 0;
			outp_east_ff <= 0;
			outp_south_ff <= 0;
		end
		else begin
			result_ff <= result;
			outp_east_ff <= inp_west;
			outp_south_ff <= inp_north;
		end
	end
	assign result = (clear_TPU==1)? 0 :result_ff + multi;
	assign outp_east = outp_east_ff;
	assign outp_south = outp_south_ff;
	assign multi = $signed(inp_north)*$signed(inp_west);
endmodule



module SystolicArray(clk, rst_n, clear_TPU, inp_west0, inp_west4, inp_west8, inp_west12,
		      inp_north0, inp_north1, inp_north2, inp_north3,
              result0, result1, result2, result3);
    input [7:0] inp_west0, inp_west4, inp_west8, inp_west12,
		      inp_north0, inp_north1, inp_north2, inp_north3;
    output [127:0]result0, result1, result2, result3;
	input clk, rst_n;
	reg [3:0] count;
	input clear_TPU;
	
	
	wire [7:0] inp_north0, inp_north1, inp_north2, inp_north3;
	wire [7:0] inp_west0, inp_west4, inp_west8, inp_west12;
	wire [7:0] outp_south0, outp_south1, outp_south2, outp_south3, outp_south4, outp_south5, outp_south6, outp_south7, outp_south8, outp_south9, outp_south10, outp_south11, outp_south12, outp_south13, outp_south14, outp_south15;
	wire [7:0] outp_east0, outp_east1, outp_east2, outp_east3, outp_east4, outp_east5, outp_east6, outp_east7, outp_east8, outp_east9, outp_east10, outp_east11, outp_east12, outp_east13, outp_east14, outp_east15;
	wire [31:0] result[15:0];
	
	//from north and west
	PE P0 (inp_north0, inp_west0, clk, rst_n, clear_TPU, outp_south0, outp_east0, result[0]);
	//from north
	PE P1 (inp_north1, outp_east0, clk, rst_n, clear_TPU,outp_south1, outp_east1, result[1]);
	PE P2 (inp_north2, outp_east1, clk, rst_n, clear_TPU,outp_south2, outp_east2, result[2]);
	PE P3 (inp_north3, outp_east2, clk, rst_n, clear_TPU,outp_south3, outp_east3, result[3]);
	
	// //from west
	PE P4 (outp_south0, inp_west4, clk, rst_n, clear_TPU, outp_south4, outp_east4, result[4]);
	PE P8 (outp_south4, inp_west8, clk, rst_n, clear_TPU,outp_south8, outp_east8, result[8]);
	PE P12 (outp_south8, inp_west12, clk, rst_n,clear_TPU, outp_south12, outp_east12, result[12]);
	
	//no direct inputs
	//second row
	PE P5 (outp_south1, outp_east4, clk, rst_n,clear_TPU, outp_south5, outp_east5, result[5]);
	PE P6 (outp_south2, outp_east5, clk, rst_n, clear_TPU,outp_south6, outp_east6, result[6]);
	PE P7 (outp_south3, outp_east6, clk, rst_n, clear_TPU,outp_south7, outp_east7, result[7]);
	//third row
	PE P9 (outp_south5, outp_east8, clk, rst_n, clear_TPU,outp_south9, outp_east9, result[9]);
	PE P10 (outp_south6, outp_east9, clk, rst_n, clear_TPU,outp_south10, outp_east10, result[10]);
	PE P11 (outp_south7, outp_east10, clk, rst_n, clear_TPU,outp_south11, outp_east11, result[11]);
	//fourth row
	PE P13 (outp_south9, outp_east12, clk, rst_n, clear_TPU, outp_south13, outp_east13, result[13]);
	PE P14 (outp_south10, outp_east13, clk, rst_n, clear_TPU, outp_south14, outp_east14, result[14]);
	PE P15 (outp_south11, outp_east14, clk, rst_n, clear_TPU, outp_south15, outp_east15, result[15]);

	assign result0 = {result[0], result[1], result[2], result[3]};
	// assign result1 = {result[4], result[5], result[6], result[7]};
	// assign result2 = {result[8], result[9], result[10], result[11]};
	// assign result3 = {result[12], result[13], result[14], result[15]};
  assign result1 = 0;
  assign result2 = 0;
  assign result3 = 0;
endmodule
