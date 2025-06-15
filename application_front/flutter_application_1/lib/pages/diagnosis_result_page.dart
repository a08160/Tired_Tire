import 'dart:io';
import 'package:flutter/material.dart';

class DiagnosisResultPage extends StatelessWidget {
  final Map<String, dynamic> result;
  final String imagePath;

  DiagnosisResultPage({required this.result, required this.imagePath});

  @override
  Widget build(BuildContext context) {
    double airPct = (result['air_pct'] ?? 0).toDouble();
    int score = airPct.round();

    String statusText;
    String commentText;
    Color statusColor;
    Color commentBgColor;
    IconData statusIcon;

    if (score >= 80) {
      statusText = "양호";
      commentText = "공기압 상태가 양호해요!";
      statusColor = Colors.green;
      commentBgColor = Color(0xFFE6F4E9);
      statusIcon = Icons.verified;
    } else if (score >= 60) {
      statusText = "주의";
      commentText = "공기주입이 필요해요!";
      statusColor = Colors.orange;
      commentBgColor = Color(0xFFFFF7E0);
      statusIcon = Icons.warning_amber_rounded;
    } else {
      statusText = "위험";
      commentText = "즉시 점검이 필요해요!";
      statusColor = Colors.red;
      commentBgColor = Color(0xFFFFEBEB);
      statusIcon = Icons.cancel;
    }

    return Scaffold(
      appBar: AppBar(title: Text("공기압 진단 결과")),
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          children: [
            Container(
              padding: EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              decoration: BoxDecoration(
                color: statusColor,
                borderRadius: BorderRadius.circular(30),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(statusIcon, color: Colors.white, size: 20),
                  SizedBox(width: 8),
                  Text(
                    statusText,
                    style: TextStyle(color: Colors.white, fontSize: 16),
                  ),
                ],
              ),
            ),
            SizedBox(height: 20),

            Text.rich(
              TextSpan(
                text: "공기압 점수 ",
                children: [
                  TextSpan(
                    text: "$score점",
                    style: TextStyle(
                      color: statusColor,
                      fontWeight: FontWeight.bold,
                      fontSize: 22,
                    ),
                  ),
                ],
              ),
              style: TextStyle(fontSize: 20),
            ),
            SizedBox(height: 10),

            Text(
              commentText,
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 30),

            _buildResultImage(), // ✅ 이 부분이 핵심

            SizedBox(height: 30),

            Container(
              width: double.infinity,
              padding: EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: commentBgColor,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text("AI 코멘트", style: TextStyle(fontWeight: FontWeight.bold)),
                  SizedBox(height: 8),
                  Text(_generateAIComment(score)),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  /// 여기서 File 이미지 경로 사용
  Widget _buildResultImage() {
    return ClipRRect(
      borderRadius: BorderRadius.circular(16),
      child: Image.file(
        File(imagePath),
        width: 240,
        height: 240,
        fit: BoxFit.cover,
      ),
    );
  }

  String _generateAIComment(int score) {
    if (score >= 80) {
      return "타이어 공기압 점수가 ${score}점으로 양호한 상태입니다.";
    } else if (score >= 60) {
      return "타이어 공기압 점수가 ${score}점으로 공기주입이 필요합니다.";
    } else {
      return "타이어 공기압 점수가 ${score}점으로 위험합니다. 즉시 점검해주세요.";
    }
  }
}
