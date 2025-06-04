import 'package:flutter/material.dart';
import 'dart:async';

class SignUpPage extends StatefulWidget {
  @override
  _SignUpPageState createState() => _SignUpPageState();
}

class _SignUpPageState extends State<SignUpPage> {
  final _nameController = TextEditingController();
  final _birthController = TextEditingController();
  final _phoneController = TextEditingController();
  final _nicknameController = TextEditingController();
  final _authCodeController = TextEditingController();

  String? _gender;
  String? _carrier;
  bool _showAuthField = false;
  int _remainingSeconds = 0;
  Timer? _timer;
  bool _isAuthVerified = false;

  final String _expectedAuthCode = "123456"; // 예시 인증번호

  final List<String> _carrierOptions = [
    'SKT',
    'KT',
    'LG U+',
    'SKT 알뜰폰',
    'KT 알뜰폰',
    'LG U+ 알뜰폰',
  ];

  void _showCarrierPicker() {
    showModalBottomSheet(
      context: context,
      builder: (context) {
        return ListView(
          children:
              _carrierOptions.map((carrier) {
                return ListTile(
                  title: Text(carrier),
                  onTap: () {
                    setState(() {
                      _carrier = carrier;
                    });
                    Navigator.pop(context);
                  },
                );
              }).toList(),
        );
      },
    );
  }

  void _startAuthTimer() {
    setState(() {
      _showAuthField = true;
      _remainingSeconds = 300;
      _isAuthVerified = false;
    });
    _timer?.cancel();
    _timer = Timer.periodic(Duration(seconds: 1), (timer) {
      if (_remainingSeconds > 0) {
        setState(() {
          _remainingSeconds--;
        });
      } else {
        timer.cancel();
      }
    });
  }

  void _verifyAuthCode() {
    if (_authCodeController.text.trim() == _expectedAuthCode &&
        _remainingSeconds > 0) {
      setState(() {
        _isAuthVerified = true;
      });
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('인증 성공!')));
    } else {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('인증 실패. 인증번호를 다시 확인해주세요.')));
    }
  }

  String _formatTime(int seconds) {
    final minutes = seconds ~/ 60;
    final secs = seconds % 60;
    return '${minutes.toString().padLeft(2, '0')}:${secs.toString().padLeft(2, '0')}';
  }

  void _showCompleteDialog() {
    if (_nameController.text.trim().isEmpty ||
        _gender == null ||
        _birthController.text.trim().length != 8 ||
        _carrier == null ||
        _phoneController.text.trim().length < 10 ||
        _nicknameController.text.trim().isEmpty ||
        !_isAuthVerified) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('모든 필드를 올바르게 입력하고 인증을 완료해주세요.')));
      return;
    }

    showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          backgroundColor: Colors.white,
          title: Text(
            '회원가입 완료! 로그인해주세요.',
            style: TextStyle(color: Colors.black),
          ),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.of(context).popUntil((route) => route.isFirst);
              },
              child: Text('확인'),
            ),
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color(0xFF1A171D),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: Text('회원가입', style: TextStyle(fontWeight: FontWeight.bold)),
      ),
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildInputField('이름', _nameController),
              SizedBox(height: 16),
              Text('성별', style: TextStyle(color: Colors.white)),
              Row(
                children:
                    ['남', '여'].map((g) {
                      final selected = _gender == g;
                      return Expanded(
                        child: GestureDetector(
                          onTap: () => setState(() => _gender = g),
                          child: Container(
                            margin: EdgeInsets.symmetric(
                              horizontal: 5,
                              vertical: 10,
                            ),
                            padding: EdgeInsets.symmetric(vertical: 12),
                            decoration: BoxDecoration(
                              color: selected ? Colors.white : Colors.white24,
                              borderRadius: BorderRadius.circular(30),
                            ),
                            alignment: Alignment.center,
                            child: Text(
                              g,
                              style: TextStyle(
                                color: selected ? Colors.black : Colors.white,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ),
                        ),
                      );
                    }).toList(),
              ),
              _buildInputField('생년월일 (예: 19900101)', _birthController),
              SizedBox(height: 16),
              Text('통신사', style: TextStyle(color: Colors.white)),
              GestureDetector(
                onTap: _showCarrierPicker,
                child: Container(
                  width: double.infinity,
                  padding: EdgeInsets.symmetric(vertical: 14, horizontal: 16),
                  margin: EdgeInsets.symmetric(vertical: 10),
                  decoration: BoxDecoration(
                    color: Colors.white24,
                    borderRadius: BorderRadius.circular(30),
                  ),
                  child: Text(
                    _carrier ?? '통신사를 선택하세요',
                    style: TextStyle(
                      color: _carrier == null ? Colors.white54 : Colors.white,
                    ),
                  ),
                ),
              ),
              SizedBox(height: 16),
              Text('전화번호', style: TextStyle(color: Colors.white)),
              SizedBox(height: 8),
              Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _phoneController,
                      style: TextStyle(color: Colors.white),
                      decoration: InputDecoration(
                        filled: true,
                        fillColor: Colors.white24,
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(30),
                          borderSide: BorderSide.none,
                        ),
                        contentPadding: EdgeInsets.symmetric(
                          horizontal: 16,
                          vertical: 14,
                        ),
                      ),
                    ),
                  ),
                  SizedBox(width: 8),
                  ElevatedButton(
                    onPressed: _startAuthTimer,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.white,
                      foregroundColor: Colors.black,
                      padding: EdgeInsets.symmetric(
                        horizontal: 12,
                        vertical: 12,
                      ),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(30),
                      ),
                    ),
                    child: Text('인증번호 발송'),
                  ),
                ],
              ),
              if (_showAuthField) ...[
                SizedBox(height: 10),
                Row(
                  children: [
                    Expanded(
                      child: TextField(
                        controller: _authCodeController,
                        style: TextStyle(color: Colors.white),
                        decoration: InputDecoration(
                          filled: true,
                          fillColor: Colors.white24,
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(30),
                            borderSide: BorderSide.none,
                          ),
                          contentPadding: EdgeInsets.symmetric(
                            horizontal: 16,
                            vertical: 14,
                          ),
                          hintText:
                              _remainingSeconds == 0
                                  ? '제한시간 초과. 인증번호를 재발송해주세요'
                                  : '인증 번호 입력',
                          hintStyle: TextStyle(color: Colors.white54),
                        ),
                      ),
                    ),
                    if (_remainingSeconds > 0) ...[
                      SizedBox(width: 10),
                      Text(
                        _formatTime(_remainingSeconds),
                        style: TextStyle(color: Colors.white),
                      ),
                    ],
                    TextButton(
                      onPressed: _verifyAuthCode,
                      child: Text('확인', style: TextStyle(color: Colors.white)),
                    ),
                  ],
                ),
              ],
              SizedBox(height: 16),
              _buildInputField('닉네임', _nicknameController),
              SizedBox(height: 30),
              ElevatedButton(
                onPressed: _showCompleteDialog,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.white,
                  foregroundColor: Colors.black,
                  minimumSize: Size(double.infinity, 48),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(30),
                  ),
                ),
                child: Text('회원가입 완료'),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildInputField(String label, TextEditingController controller) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: TextStyle(color: Colors.white)),
        SizedBox(height: 8),
        TextField(
          controller: controller,
          style: TextStyle(color: Colors.white),
          decoration: InputDecoration(
            filled: true,
            fillColor: Colors.white24,
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(30),
              borderSide: BorderSide.none,
            ),
            contentPadding: EdgeInsets.symmetric(horizontal: 16, vertical: 14),
          ),
        ),
      ],
    );
  }

  @override
  void dispose() {
    _nameController.dispose();
    _birthController.dispose();
    _phoneController.dispose();
    _nicknameController.dispose();
    _authCodeController.dispose();
    _timer?.cancel();
    super.dispose();
  }
}
