package MTN;


import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

//更新顺序U,S,T,a,b,c


public class MTN_adam extends InitTensor{

	public double sumTime = 0; //训练累计时间

	public int tr = 0;  //统计本轮迭代与前一轮小于误差的次数

	public int threshold = 0;  //连续下降轮数小于误差范围切达到阈值终止训练

	public boolean flagRMSE = true, flagMAE = true;

	public String str = null;

	public double yita = 0;
	public double gama = 0;
	public double lambda = 0;  //因子矩阵正则化参数
	public double lambda_b = 0;  //线性偏差正则化参数

	public double maxRR = -100;
	public int minRRRound = 0;
	public double[] timeCost = new double[500+1];

	public double beta1, beta2;
	public double error;
	public double alpha;

	MTN_adam(String trainFile, String validFile, String testFile, String separator )
	{
		super(trainFile, validFile, testFile, separator); 
	}

	public void train(FileWriter Parameters) throws IOException
	{
		long startTime = System.currentTimeMillis();   //记录开始训练时间
		
//		FileWriter  fw = new FileWriter(new File(trainFile.replace(".txt", "_")+rank+"_"+lambda+"_"+
//		                lambda_b+"_"+new Date().getTime() / 1000+"_BNLFT.txt"));
//
//		fw.write("round :: everyRoundRMSE :: everyRoundMAE :: costTime(ms) \n");
//		fw.flush();
		
		initFactorMatrix();
		initAssistMatrix();
//		initSliceSet();

//		System.out.println("maxAID maxBID maxCID "+maxAID+" "+maxBID+" "+maxCID);
//		System.out.println("minAID minBID minCID "+minAID+" "+minBID+" "+minCID);
//		System.out.println("trainCount validCount testCount "+trainCount+" "+validCount+" "+testCount);
//		System.out.println("初始范围:"+initscale);
//		System.out.println("lambda  lambda_b: "+lambda+" "+lambda_b);
		
		
		for(int round = 1; round <= trainRound; round++)
		{
			long startRoundTime = System.currentTimeMillis();    //记录每轮的训练时间
			
//			initAssistMatrix();
			
			for(TensorTuple trainTuple: trainData)
			{
				trainTuple.valueHat = this.getPrediction(trainTuple.aID,trainTuple.bID, trainTuple.cID);

				double err = trainTuple.value-trainTuple.valueHat;
				for(int r1 = 1; r1 <= rank; r1++) {
					for(int r4 = 1; r4 <= rank1; r4++) {
						double temp = 0;
						for (int r2 = 1; r2 <= rank; r2++) {
							for (int r3 = 1; r3 <= rank; r3++) {
								temp += G[r1][r2][r3] * D[trainTuple.bID][r2][r4] * T[trainTuple.cID][r3];
							}
						}
						double gradient_S = -err * temp + lambda*S[trainTuple.aID][r1][r4];
						Sup[trainTuple.aID][r1][r4] = beta1*Sup[trainTuple.aID][r1][r4]+(1-beta1)*gradient_S;
						Sdown[trainTuple.aID][r1][r4] = beta2*Sdown[trainTuple.aID][r1][r4]+(1-beta2)*Math.pow(gradient_S, 2);

						double hat_Sup = Sup[trainTuple.aID][r1][r4]/(1-Math.pow(beta1, round));
						double hat_Sdown = Sdown[trainTuple.aID][r1][r4]/(1-Math.pow(beta2, round));

						S[trainTuple.aID][r1][r4] = S[trainTuple.aID][r1][r4]
								-alpha*hat_Sup/(Math.sqrt(hat_Sdown+error));
					}
				}


				trainTuple.valueHat = this.getPrediction(trainTuple.aID,trainTuple.bID, trainTuple.cID);
				err = trainTuple.value-trainTuple.valueHat;
				for(int r2 = 1; r2 <= rank; r2++) {
					for(int r4 = 1; r4 <= rank1; r4++) {
						double temp = 0;
						for (int r1 = 1; r1 <= rank; r1++) {
							for (int r3 = 1; r3 <= rank; r3++) {
								temp += G[r1][r2][r3] * S[trainTuple.aID][r1][r4] * T[trainTuple.cID][r3];
							}
						}
						double gradient_D = -err * temp+lambda*D[trainTuple.bID][r2][r4];
						Dup[trainTuple.bID][r2][r4] = beta1*Dup[trainTuple.bID][r2][r4]+(1-beta1)*gradient_D;
						Ddown[trainTuple.bID][r2][r4] = beta2*Ddown[trainTuple.bID][r2][r4]+(1-beta2)*Math.pow(gradient_D, 2);

						double hat_Dup = Dup[trainTuple.bID][r2][r4]/(1-Math.pow(beta1, round));
						double hat_Ddown = Ddown[trainTuple.bID][r2][r4]/(1-Math.pow(beta2, round));

						D[trainTuple.bID][r2][r4] = D[trainTuple.bID][r2][r4]
								-alpha*hat_Dup/(Math.sqrt(hat_Ddown+error));
					}
				}

				trainTuple.valueHat = this.getPrediction(trainTuple.aID,trainTuple.bID, trainTuple.cID);
				err = trainTuple.value-trainTuple.valueHat;

				for(int r3 = 1; r3 <= rank; r3++) {
					double temp = 0;
					for (int r1 = 1; r1 <= rank; r1++) {
						for (int r2 = 1; r2 <= rank; r2++) {
							for (int r4 = 1; r4 <= rank1; r4++) {
								temp += G[r1][r2][r3] * D[trainTuple.bID][r2][r4] * S[trainTuple.aID][r1][r4];
							}
						}
					}
					double gradient_T = -err * temp+lambda*T[trainTuple.cID][r3];
					Tup[trainTuple.cID][r3] = beta1*Tup[trainTuple.cID][r3]+(1-beta1)*gradient_T;
					Tdown[trainTuple.cID][r3] = beta2*Tdown[trainTuple.cID][r3]+(1-beta2)*Math.pow(gradient_T, 2);

					double hat_Tup= Tup[trainTuple.cID][r3]/(1-Math.pow(beta1, round));
					double hat_Tdown = Tdown[trainTuple.cID][r3]/(1-Math.pow(beta2, round));

					T[trainTuple.cID][r3] = T[trainTuple.cID][r3]
							-alpha*hat_Tup/(Math.sqrt(hat_Tdown+error));
				}

				trainTuple.valueHat = this.getPrediction(trainTuple.aID,trainTuple.bID, trainTuple.cID);
				err = trainTuple.value-trainTuple.valueHat;

				for(int r1 = 1; r1 <= rank; r1++) {
					for (int r2 = 1; r2 <= rank; r2++) {
						for (int r3 = 1; r3 <= rank; r3++) {
							double temp = 0;
							for (int r4 = 1; r4 <= rank1; r4++) {
								temp += S[trainTuple.aID][r1][r4] * D[trainTuple.bID][r2][r4] * T[trainTuple.cID][r3];
							}
							double gradient_G = -err * temp+lambda*G[r1][r2][r3];
							Gup[r1][r2][r3] = beta1*Gup[r1][r2][r3]+(1-beta1)*gradient_G;
							Gdown[r1][r2][r3] = beta2*Gdown[r1][r2][r3]+(1-beta2)*Math.pow(gradient_G, 2);

							double hat_Gup = Gup[r1][r2][r3]/(1-Math.pow(beta1, round));
							double hat_Gdown = Gdown[r1][r2][r3]/(1-Math.pow(beta2, round));

							G[r1][r2][r3] = G[r1][r2][r3]
									-alpha*hat_Gup/(Math.sqrt(hat_Gdown+error));
						}
					}
				}

				trainTuple.valueHat = this.getPrediction(trainTuple.aID,trainTuple.bID, trainTuple.cID);
				 err = trainTuple.value-trainTuple.valueHat;

				double gradient_a = -err+lambda*a[trainTuple.aID];
				aup[trainTuple.aID] = beta1*aup[trainTuple.aID]+(1-beta1)*gradient_a;
				adown[trainTuple.aID] = beta2*adown[trainTuple.aID]+(1-beta2)*Math.pow(gradient_a, 2);

				double hat_aup = aup[trainTuple.aID]/(1-Math.pow(beta1, round));
				double hat_adown = adown[trainTuple.aID]/(1-Math.pow(beta2, round));

				a[trainTuple.aID] = a[trainTuple.aID]
						-alpha*hat_aup/(Math.sqrt(hat_adown+error));


				trainTuple.valueHat = this.getPrediction(trainTuple.aID,trainTuple.bID, trainTuple.cID);
				err = trainTuple.value-trainTuple.valueHat;

				double gradient_b = -err+lambda*b[trainTuple.bID];
				bup[trainTuple.bID] = beta1*bup[trainTuple.bID]+(1-beta1)*gradient_b;
				bdown[trainTuple.bID] = beta2*bdown[trainTuple.bID]+(1-beta2)*Math.pow(gradient_b, 2);

				double hat_bup = bup[trainTuple.bID]/(1-Math.pow(beta1, round));
				double hat_bdown = bdown[trainTuple.bID]/(1-Math.pow(beta2, round));

				b[trainTuple.bID] = b[trainTuple.bID]
						-alpha*hat_bup/(Math.sqrt(hat_bdown+error));


				trainTuple.valueHat = this.getPrediction(trainTuple.aID,trainTuple.bID, trainTuple.cID);
				err = trainTuple.value-trainTuple.valueHat;

				double gradient_c = -err+lambda*c[trainTuple.cID];
				cup[trainTuple.cID] = beta1*cup[trainTuple.cID]+(1-beta1)*gradient_c;
				cdown[trainTuple.cID] = beta2*cdown[trainTuple.cID]+(1-beta2)*Math.pow(gradient_c, 2);

				double hat_cup = cup[trainTuple.cID]/(1-Math.pow(beta1, round));
				double hat_cdown = cdown[trainTuple.cID]/(1-Math.pow(beta2, round));

				c[trainTuple.cID] = c[trainTuple.cID]
						-alpha*hat_cup/(Math.sqrt(hat_cdown+error));

			}



			timeCost[0] = 0;
			timeCost[round] = timeCost[round-1] + (System.currentTimeMillis() - startRoundTime)/1000.00;
			
			// 每一轮参数更新后，开始对验证集测试
			double square = 0, absCount = 0;
			for (TensorTuple validTuple : validData) {
				// 获得元素的预测值
				validTuple.valueHat = this.getPrediction(validTuple.aID, validTuple.bID, validTuple.cID);
				square += Math.pow(validTuple.value - validTuple.valueHat, 2);
				absCount += Math.abs(validTuple.value - validTuple.valueHat);
				
			}
			
			everyRoundRMSE[round] = Math.sqrt(square / validCount);
			everyRoundMAE[round] = absCount / validCount; 
			
//			long endRoundTime = System.currentTimeMillis();
//			sumTime += (endRoundTime-startRoundTime);
			
			// 每一轮参数更新后，记录测试集结果
			double square2 = 0, absCount2 = 0, absCount33 = 0, absCount44 = 0;
			for (TensorTuple testTuple : testData) {
				// 获得元素的预测值
				testTuple.valueHat = this.getPrediction(testTuple.aID, testTuple.bID, testTuple.cID);
				square2 += Math.pow(testTuple.value - testTuple.valueHat, 2);
				absCount2 += Math.abs(testTuple.value - testTuple.valueHat);
//				absCount33 += (Math.abs(testTuple.value - testTuple.valueHat)) / (testTuple.value+0.01);
				absCount44 += Math.pow( testSum / testCount - testTuple.value, 2);
			}

			everyRoundRMSE2[round] = Math.sqrt(square2 / testCount);
			everyRoundMAE2[round] = absCount2 / testCount;
//			everyRoundMAPE2[round] = 100 * absCount33 / testCount;
			everyRoundRR2[round] = 1 - square2/absCount44;

			System.out.println(everyRoundRMSE2[round] + " " + everyRoundMAE2[round] + " " + everyRoundRR2[round]);
			
//			fw.write(round + "::" + everyRoundRMSE[round] + "::" + everyRoundMAE[round]
//					+"::"+ (endRoundTime-startRoundTime)+"::"+ everyRoundRMSE2[round] + "::" + everyRoundMAE2[round]+"\n");
//			fw.flush();

			if(maxRR < everyRoundRR2[round])
			{
				maxRR = everyRoundRR2[round];
				minRRRound = round;
			}

			if (everyRoundRMSE[round-1] - everyRoundRMSE[round] > errorgap) 
			{
				if(minRMSE > everyRoundRMSE[round])
				{
					minRMSE = everyRoundRMSE[round];
					minRMSERound = round;
				}
				
				flagRMSE = false;
				tr = 0;
			}
			
			if (everyRoundMAE[round-1] - everyRoundMAE[round] > errorgap) 
			{
				if(minMAE > everyRoundMAE[round])
				{
					minMAE = everyRoundMAE[round];
					minMAERound = round;
				}
				
				flagMAE = false;
				tr = 0;
			} 
		
			if(flagRMSE && flagMAE)
			{
				tr++;
				if(tr == threshold)                
					break;
			}
			
			flagRMSE = true;
			flagMAE = true;
			
		}
		
		long endTime = System.currentTimeMillis();

//		fw.write("总训练时间："+(endTime-startTime)/1000+"s\n");
//		fw.flush();
//		fw.write("validation minRMSE:"+minRMSE+"  minRSMERound"+minRMSERound+"\n");
//		fw.write("validation minMAE:"+minMAE+"  minMAERound"+minMAERound+"\n");
//		fw.write("testing minRMSE:"+everyRoundRMSE2[minRMSERound]+"  minRSMERound"+minRMSERound+"\n");
//		fw.write("testing minMAE:"+everyRoundMAE2[minMAERound]+"  minMAERound"+minMAERound+"\n");
//		fw.flush();
//		fw.write("rank="+rank+"\n");
//		fw.flush();
//		fw.write("trainCount: "+trainCount+"validCount: "+validCount+"   testCount: "+testCount+"\n");
//		fw.flush();
//        SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");//设置日期格式
//        fw.write("训练时间："+df.format(new Date())+"\n");// new Date()为获取当前系统时间
//        fw.flush();
//		fw.write("初始范围"+initscale+"\n");
//		fw.flush();
//		fw.write("maxAID maxBID maxCID "+maxAID+" "+maxBID+" "+maxCID+"\n");
//		fw.write("minAID minBID minCID "+minAID+" "+minBID+" "+minCID+"\n");
//		fw.close();

		Parameters.write("testing minRMSE:" + everyRoundRMSE2[minRMSERound] + " minRSMERound::" + minRMSERound + " RMSECostTime::" +timeCost[minRMSERound]/60+"min\n");
		Parameters.write("testing minMAE:" + everyRoundMAE2[minMAERound]+ " minMAERound:" + minMAERound + " MAECostTime::"+timeCost[minMAERound]/60+"min\n");
//		Parameters.write("testing minMAPE:" + everyRoundMAPE2[minMAPERound]+ " minMAPERound::" + minMAPERound + " MAPECostTime::" + "\n");
		Parameters.write("testing minRR:" + everyRoundRR2[minRRRound]+ " minRRRound::" + minRRRound + " RRCostTime::"+timeCost[minRMSERound]/60+"min\n");
		Parameters.write( "init::"+ initscale+"--"+ yita + "::lambda::"+ lambda +"timeCost::"+ timeCost[Math.max(minRMSERound,minMAERound)]/60+ "min\n");
		Parameters.write( "------------------------------------------------------------------------------------------------" + "\n");
		Parameters.flush();

		System.out.println("***********************************************");
		System.out.println("rank: "+this.rank+"\n");
//		System.out.println("validation minRMSE:"+minRMSE+"  minRSMERound"+minRMSERound);
//		System.out.println("validation minMAE:"+minMAE+"  minMAERound"+minMAERound);
		System.out.println("总训练时间:"+(endTime-startTime)/60000.00+"min\n");
		System.out.println("训练时间:"+ timeCost[Math.max(minRMSERound,minMAERound)]/60+"min\n");
		System.out.println("testing minRMSE:"+everyRoundRMSE2[minRMSERound]+"  minRSMERound"+minRMSERound);
		System.out.println("testing minMAE:"+everyRoundMAE2[minMAERound]+"  minMAERound"+minMAERound);

	}

	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		long S = System.currentTimeMillis();
		for (int number = 1; number <= 8; number++) {

//			double[] Parameter1 = new double[]{0,5,6,6,8,9,8,7,7};
//			double[] Parameter2 = new double[]{0,8,8,9,10,7,7,6,6};
			FileWriter Parameters = new FileWriter(new File("D:\\Tensor\\R=5\\MTN-adam\\parameter" + number + ".txt"));
			for (int i = 0; i <= 12; i++) {
				for (int j = 0; j <= 12; j++) {
					//训练：验证：测试=5%：5%：90%
					MTN_adam bnlft = new MTN_adam(
							"D:\\Tensor\\Dataset\\network\\D" + number + "\\0.1tr.txt",
							"D:\\Tensor\\Dataset\\network\\D" + number + "\\0.2va.txt",
							"D:\\Tensor\\Dataset\\network\\D" + number + "\\0.7te.txt", "::");

//					bnlft.initscale = 0.3+0.01*i;
					if (number <= 4) {
						bnlft.initscale = 0.14;
						bnlft.initscale2 = 0.04;
					}
					else {
						bnlft.initscale = 0.3;
						bnlft.initscale2 = 0.07;
					}

//					bnlft.initscale2 = 0.04;
					bnlft.threshold = 2;
					bnlft.rank = 5;
					bnlft.rank1 = 5;
					bnlft.trainRound = 500;
					bnlft.errorgap = 1E-5;
//					bnlft.yita = Math.pow(2, (-Parameter1[number]));
//					bnlft.gama = 1;
					bnlft.lambda = Math.pow(2, -i);     //根据数据集设置
//					bnlft.lambda = 0;     //根据数据集设置

					bnlft.alpha = Math.pow(2, -j);
					bnlft.beta1 = 0.9;
					bnlft.beta2 = 0.999;
					bnlft.error = 0.00001;

					try {
						bnlft.initData(bnlft.trainFile, bnlft.trainData, 1);
						bnlft.initData(bnlft.validFile, bnlft.validData, 2);
						bnlft.initData(bnlft.testFile, bnlft.testData, 3);
						bnlft.train(Parameters);
					} catch (IOException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
			}
			long E = System.currentTimeMillis();
			Parameters.write( "TotalTime is: " + (E - S)/60 + "\n");
			Parameters.flush();
		}
	}
}

