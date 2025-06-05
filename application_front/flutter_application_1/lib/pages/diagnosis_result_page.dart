import 'package:flutter/material.dart';

class DiagnosisResultPage extends StatelessWidget {
  final Map<String, dynamic> result;

  DiagnosisResultPage({required this.result});

  @override
  Widget build(BuildContext context) {
    String displayText;
    if (result['success'] == true) {
      displayText = "공기압 상태: ${result['air_pct']}%";
    } else {
      displayText = "오류 발생: ${result['message']}";
    }

    return Scaffold(
      appBar: AppBar(title: Text("진단 결과")),
      body: Center(child: Text(displayText, style: TextStyle(fontSize: 20))),
    );
  }
}
