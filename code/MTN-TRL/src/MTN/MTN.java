package MTN;


import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

//更新顺序U,S,T,a,b,c


public class MTN extends InitTensor{

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
	public double[] timeCost = new double[1000+1];

	MTN(String trainFile, String validFile, String testFile, String separator )
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

				for(int r1 = 1; r1 <= rank; r1++) {
					for(int r4 = 1; r4 <= rank1; r4++) {
						double temp = 0;
						for (int r2 = 1; r2 <= rank; r2++) {
							for (int r3 = 1; r3 <= rank; r3++) {
								temp += G[r1][r2][r3] * D[trainTuple.bID][r2][r4] * T[trainTuple.cID][r3];
							}
						}
						Sup[trainTuple.aID][r1][r4] = (trainTuple.value-trainTuple.valueHat) * temp - lambda * S[trainTuple.aID][r1][r4];
						S[trainTuple.aID][r1][r4] += yita*Sup[trainTuple.aID][r1][r4] - gama*Sdown[trainTuple.aID][r1][r4];
						Sdown[trainTuple.aID][r1][r4] = gama*Sdown[trainTuple.aID][r1][r4] - yita*Sup[trainTuple.aID][r1][r4];
					}
				}
			}
			
			
			for(TensorTuple trainTuple: trainData)
			{
				trainTuple.valueHat = this.getPrediction(trainTuple.aID,trainTuple.bID, trainTuple.cID);

				for(int r2 = 1; r2 <= rank; r2++) {
					for(int r4 = 1; r4 <= rank1; r4++) {
						double temp = 0;
						for (int r1 = 1; r1 <= rank; r1++) {
							for (int r3 = 1; r3 <= rank; r3++) {
								temp += G[r1][r2][r3] * S[trainTuple.aID][r1][r4] * T[trainTuple.cID][r3];
							}
						}
						Dup[trainTuple.bID][r2][r4] = (trainTuple.value-trainTuple.valueHat) * temp - lambda * D[trainTuple.bID][r2][r4];
						D[trainTuple.bID][r2][r4] += yita*Dup[trainTuple.bID][r2][r4] - gama*Ddown[trainTuple.bID][r2][r4];
						Ddown[trainTuple.bID][r2][r4] = gama*Ddown[trainTuple.bID][r2][r4] - yita*Dup[trainTuple.bID][r2][r4];
					}
				}
			}


			for(TensorTuple trainTuple: trainData)
			{
				trainTuple.valueHat = this.getPrediction(trainTuple.aID,trainTuple.bID, trainTuple.cID);

				for(int r3 = 1; r3 <= rank; r3++) {
					double temp = 0;
					for (int r1 = 1; r1 <= rank; r1++) {
						for (int r2 = 1; r2 <= rank; r2++) {
							for (int r4 = 1; r4 <= rank1; r4++) {
								temp += G[r1][r2][r3] * D[trainTuple.bID][r2][r4] * S[trainTuple.aID][r1][r4];
							}
						}
					}
					Tup[trainTuple.cID][r3] = (trainTuple.value-trainTuple.valueHat) * temp - lambda * T[trainTuple.cID][r3];
					T[trainTuple.cID][r3] += yita*Tup[trainTuple.cID][r3] - gama*Tdown[trainTuple.cID][r3];
					Tdown[trainTuple.cID][r3] = gama*Tdown[trainTuple.cID][r3] - yita*Tup[trainTuple.cID][r3];
				}
			}



			for(TensorTuple trainTuple: trainData)
			{
				trainTuple.valueHat = this.getPrediction(trainTuple.aID,trainTuple.bID, trainTuple.cID);

				for(int r1 = 1; r1 <= rank; r1++) {
					for (int r2 = 1; r2 <= rank; r2++) {
						for (int r3 = 1; r3 <= rank; r3++) {
							double temp = 0;
							for (int r4 = 1; r4 <= rank1; r4++) {
								temp += S[trainTuple.aID][r1][r4] * D[trainTuple.bID][r2][r4] * T[trainTuple.cID][r3];
							}
							Gup[r1][r2][r3] = (trainTuple.value-trainTuple.valueHat) * temp - lambda * G[r1][r2][r3];
							G[r1][r2][r3] += yita*Gup[r1][r2][r3] - gama*Gdown[r1][r2][r3];
							Gdown[r1][r2][r3] = gama*Gdown[r1][r2][r3] - yita*Gup[r1][r2][r3];
						}
					}
				}
			}


//			bias更新

			for(TensorTuple trainTuple: trainData)
			{
				trainTuple.valueHat = this.getPrediction(trainTuple.aID,trainTuple.bID, trainTuple.cID);
				aup[trainTuple.aID] = (trainTuple.value-trainTuple.valueHat) - lambda*a[trainTuple.aID];
				a[trainTuple.aID] += yita*aup[trainTuple.aID] - gama*adown[trainTuple.aID];
				adown[trainTuple.aID] = gama*adown[trainTuple.aID] - yita*aup[trainTuple.aID];
			}



			for(TensorTuple trainTuple: trainData)
			{
				trainTuple.valueHat = this.getPrediction(trainTuple.aID,trainTuple.bID, trainTuple.cID);
				bup[trainTuple.bID] = (trainTuple.value-trainTuple.valueHat) - lambda*b[trainTuple.bID];
				b[trainTuple.bID] += yita*bup[trainTuple.bID] - gama*bdown[trainTuple.bID];
				bdown[trainTuple.bID] = gama*bdown[trainTuple.bID] - yita*bup[trainTuple.bID];
			}



			for(TensorTuple trainTuple: trainData)
			{
				trainTuple.valueHat = this.getPrediction(trainTuple.aID,trainTuple.bID, trainTuple.cID);
				cup[trainTuple.cID] = (trainTuple.value-trainTuple.valueHat) - lambda*c[trainTuple.cID];
				c[trainTuple.cID] += yita*cup[trainTuple.cID] - gama*cdown[trainTuple.cID];
				cdown[trainTuple.cID] = gama*cdown[trainTuple.cID] - yita*cup[trainTuple.cID];
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
		Parameters.write("testing minRR:" + everyRoundRR2[minRRRound]+ " minRRRound::" + minRRRound + " RRCostTime::"+timeCost[minRRRound]/60+"min\n");
		Parameters.write( "init::"+ initscale+"--"+initscale2+"yita::"+ yita + "::lambda::"+ lambda +"timeCost::"+ timeCost[Math.max(minRMSERound,minMAERound)]/60+ "min\n");
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
		for (int number = 4; number <= 4; number++) {

			double[] Parameter1 = new double[]{0,11,11,11,11,10,11,9,12};
			double[] Parameter2 = new double[]{0,4,12,12,12,6,6,6,5};

			long S = System.currentTimeMillis();
			FileWriter Parameters = new FileWriter(new File("D:\\Tensor\\R=5\\MTN动量参数\\test1" + number + ".txt"));
			for (int i = 0; i <= 12; i++) {
				for (int j = 0; j <= 12; j++) {
					//训练：验证：测试=5%：5%：90%
					MTN bnlft = new MTN(
							"D:\\Tensor\\Dataset\\network\\D" + number + "\\0.1tr.txt",
							"D:\\Tensor\\Dataset\\network\\D" + number + "\\0.2va.txt",
							"D:\\Tensor\\Dataset\\network\\D" + number + "\\0.7te.txt", "::");

					if (number <= 4) {
						bnlft.initscale = 0.14;
						bnlft.initscale2 = 0.04;
					}
					else {
						bnlft.initscale = 0.3;
						bnlft.initscale2 = 0.07;
					}

					bnlft.threshold = 2;
					bnlft.rank = 5;
					bnlft.rank1 = 5;
					bnlft.trainRound = 1000;
					bnlft.errorgap = 1E-5;
					bnlft.yita = Math.pow(2, (-i));
					bnlft.gama = 0.9;
					bnlft.lambda = Math.pow(2, (-j));     //根据数据集设置

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

